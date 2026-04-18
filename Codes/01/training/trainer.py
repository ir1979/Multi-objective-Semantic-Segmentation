"""Enhanced training module for building footprint segmentation.

This module provides comprehensive training functionality with support for:
- Configuration-based training (YAML)
- Automatic batch size adjustment for GPU memory
- Multiple optimizers and schedulers
- Early stopping and checkpointing
- Integration with MOO evaluation framework
- Training curve visualization
"""

import os
import glob
import time
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (
    CSVLogger, ModelCheckpoint, TensorBoard, 
    EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
)
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.optimizers.schedules import CosineDecay

# Import framework modules
from models.model_factory import build_model, get_model_info
from losses.loss_manager import build_single_loss, build_losses
from training.metrics import (
    iou_score, dice_score, precision_score, recall_score,
    pixel_accuracy, boundary_iou, boundary_f1, compactness_score, topological_correctness
)
from data.dataset import BuildingDataset, create_datasets_from_config
from utils.repro import set_global_seed, get_git_commit, save_json
from utils.system import get_process_memory_mb

logger = logging.getLogger(__name__)


def get_optimizer(optimizer_name: str, learning_rate: float, **kwargs) -> keras.optimizers.Optimizer:
    """Get optimizer by name.
    
    Parameters
    ----------
    optimizer_name : str
        Name of optimizer ('adam', 'sgd')
    learning_rate : float
        Learning rate
    **kwargs
        Additional optimizer parameters
    
    Returns
    -------
    keras.optimizers.Optimizer
        Configured optimizer
    """
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == 'adam':
        return Adam(
            learning_rate=learning_rate,
            beta_1=kwargs.get('beta_1', 0.9),
            beta_2=kwargs.get('beta_2', 0.999),
            epsilon=kwargs.get('epsilon', 1e-7)
        )
    elif optimizer_name == 'sgd':
        return SGD(
            learning_rate=learning_rate,
            momentum=kwargs.get('momentum', 0.9),
            nesterov=kwargs.get('nesterov', True)
        )
    else:
        logger.warning(f"Unknown optimizer '{optimizer_name}', using Adam")
        return Adam(learning_rate=learning_rate)


def get_scheduler(
    scheduler_name: str, 
    optimizer: keras.optimizers.Optimizer,
    epochs: int,
    steps_per_epoch: int,
    **kwargs
) -> Optional[keras.callbacks.Callback]:
    """Get learning rate scheduler.
    
    Parameters
    ----------
    scheduler_name : str
        Name of scheduler ('reducelronplateau', 'cosineannealing', 'none')
    optimizer : keras.optimizers.Optimizer
        Optimizer instance
    epochs : int
        Total training epochs
    steps_per_epoch : int
        Steps per epoch
    **kwargs
        Additional scheduler parameters
    
    Returns
    -------
    Optional[keras.callbacks.Callback]
        Scheduler callback or None
    """
    if scheduler_name is None or scheduler_name.lower() == 'none':
        return None
    
    scheduler_name = scheduler_name.lower().replace('_', '').replace('-', '')
    
    if scheduler_name == 'reducelronplateau':
        return ReduceLROnPlateau(
            monitor='val_loss',
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 5),
            min_lr=kwargs.get('min_lr', 1e-7),
            verbose=1
        )
    elif scheduler_name in ['cosineannealing', 'cosinedecay', 'cosine']:
        total_steps = epochs * steps_per_epoch
        lr_schedule = CosineDecay(
            initial_learning_rate=optimizer.learning_rate.numpy(),
            decay_steps=total_steps,
            alpha=kwargs.get('alpha', 0.0)  # Minimum learning rate fraction
        )
        optimizer.learning_rate = lr_schedule
        return None  # CosineDecay is built into optimizer
    else:
        logger.warning(f"Unknown scheduler '{scheduler_name}', no scheduler used")
        return None


def auto_adjust_batch_size(
    model_builder: Callable,
    input_shape: Tuple[int, int, int],
    start_batch_size: int = 16,
    min_batch_size: int = 1
) -> int:
    """Automatically find the maximum batch size that fits in GPU memory.
    
    Parameters
    ----------
    model_builder : Callable
        Function that builds the model
    input_shape : Tuple[int, int, int]
        Input shape (H, W, C)
    start_batch_size : int
        Starting batch size to try
    min_batch_size : int
        Minimum acceptable batch size
    
    Returns
    -------
    int
        Maximum batch size that fits in memory
    """
    batch_size = start_batch_size
    
    while batch_size >= min_batch_size:
        try:
            # Clear any existing models
            tf.keras.backend.clear_session()
            
            # Build model
            model = model_builder()
            
            # Create dummy data
            dummy_input = tf.random.normal((batch_size,) + input_shape)
            dummy_target = tf.random.uniform((batch_size,) + input_shape[:2] + (1,), minval=0, maxval=2, dtype=tf.int32)
            dummy_target = tf.cast(dummy_target, tf.float32)
            
            # Try forward pass
            with tf.device('/GPU:0'):
                _ = model(dummy_input, training=False)
                
                # Try a simple training step
                with tf.GradientTape() as tape:
                    predictions = model(dummy_input, training=True)
                    loss = tf.reduce_mean(tf.square(predictions - dummy_target))
                
                gradients = tape.gradient(loss, model.trainable_variables)
            
            # If successful, clean up and return
            del model, dummy_input, dummy_target
            tf.keras.backend.clear_session()
            
            logger.info(f"Auto-adjusted batch size: {batch_size}")
            return batch_size
            
        except tf.errors.ResourceExhaustedError:
            logger.warning(f"Batch size {batch_size} too large for GPU memory, trying {batch_size // 2}")
            batch_size = batch_size // 2
            tf.keras.backend.clear_session()
            continue
        
        except Exception as e:
            logger.warning(f"Error testing batch size {batch_size}: {e}")
            batch_size = batch_size // 2
            tf.keras.backend.clear_session()
            continue
    
    raise RuntimeError(f"Could not find suitable batch size, even {min_batch_size} failed")


def _replicate_targets(sequence, output_count):
    """Repeat a single target across multiple output heads.
    
    When the model produces deep supervision outputs, each target mask
    needs to be replicated to match the number of predictions.
    
    Parameters
    ----------
    sequence : Sequence
        Dataset sequence
    output_count : int
        Number of output heads
    
    Yields
    ------
    Tuple
        (inputs, tuple of targets)
    """
    while True:
        for x, y in sequence:
            yield x, tuple([y] * output_count)


def _latest_checkpoint(checkpoint_dir):
    """Find the most recent saved checkpoint file in a directory.
    
    Parameters
    ----------
    checkpoint_dir : str
        Directory to search
    
    Returns
    -------
    Optional[str]
        Path to latest checkpoint or None
    """
    checkpoints = sorted(
        glob.glob(os.path.join(checkpoint_dir, "*.h5")),
        key=os.path.getmtime,
    )
    return checkpoints[-1] if checkpoints else None


def build_callbacks(
    checkpoint_dir: str,
    log_dir: str,
    run_dir: str,
    early_stopping_config: Dict[str, Any],
    scheduler_config: Dict[str, Any],
    save_best_only: bool = True
) -> List[keras.callbacks.Callback]:
    """Build training callbacks.
    
    Parameters
    ----------
    checkpoint_dir : str
        Directory for model checkpoints
    log_dir : str
        Directory for TensorBoard logs
    run_dir : str
        Run directory for CSV logs
    early_stopping_config : Dict[str, Any]
        Early stopping configuration
    scheduler_config : Dict[str, Any]
        Scheduler configuration
    save_best_only : bool
        Whether to save only best checkpoints
    
    Returns
    -------
    List[keras.callbacks.Callback]
        List of callbacks
    """
    callbacks = []
    
    # Model checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, "epoch_{epoch:03d}.h5")
    callbacks.append(ModelCheckpoint(
        checkpoint_path,
        save_weights_only=False,
        save_freq="epoch",
        save_best_only=save_best_only,
        monitor='val_iou_score' if save_best_only else None,
        mode='max',
        verbose=1
    ))
    
    # CSV logger
    callbacks.append(CSVLogger(
        os.path.join(run_dir, "logs.csv"),
        append=True
    ))
    
    # TensorBoard
    callbacks.append(TensorBoard(
        log_dir=log_dir,
        update_freq="epoch",
        histogram_freq=1
    ))
    
    # Early stopping
    if early_stopping_config.get('enabled', True):
        callbacks.append(EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_config.get('patience', 10),
            min_delta=early_stopping_config.get('min_delta', 0.001),
            restore_best_weights=True,
            verbose=1
        ))
    
    return callbacks


def train_from_config(
    config: Dict[str, Any],
    output_dir: str = "./results",
    verbose: int = 2,
    resume: bool = False,
    auto_batch_size: bool = False
) -> Dict[str, Any]:
    """Train model from configuration dictionary.
    
    This is the main training function that uses the MOO framework
    configuration format.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary containing:
        - architecture: model architecture
        - encoder: encoder backbone
        - loss_function: loss function name
        - optimizer: optimizer name
        - learning_rate: learning rate
        - batch_size: batch size
        - epochs: number of epochs
        - seed: random seed
        - tile_size: input tile size
    output_dir : str
        Output directory for results
    verbose : int
        Verbosity level
    resume : bool
        Whether to resume from checkpoint
    auto_batch_size : bool
        Whether to automatically adjust batch size
    
    Returns
    -------
    Dict[str, Any]
        Training summary dictionary
    """
    # Set random seed for reproducibility
    seed = config.get('seed', 42)
    set_global_seed(seed)
    
    # Create run directory
    experiment_name = config.get('experiment_id', f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    run_dir = os.path.join(output_dir, experiment_name)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    log_dir = os.path.join(run_dir, "tensorboard")
    
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Extract configuration
    architecture = config.get('architecture', 'Unet')
    encoder = config.get('encoder', 'resnet34')
    encoder_weights = config.get('encoder_weights', 'imagenet')
    tile_size = config.get('tile_size', 256)
    num_classes = config.get('num_classes', 1)
    deep_supervision = config.get('deep_supervision', False)
    
    input_shape = (tile_size, tile_size, 3)
    
    # Build model
    logger.info(f"Building model: {architecture} with {encoder}")
    model = build_model(
        architecture=architecture,
        encoder=encoder,
        encoder_weights=encoder_weights,
        input_shape=input_shape,
        num_classes=num_classes,
        deep_supervision=deep_supervision
    )
    
    # Get model info
    model_info = get_model_info(model)
    
    # Auto-adjust batch size if requested
    if auto_batch_size:
        batch_size = auto_adjust_batch_size(
            lambda: build_model(architecture, encoder, encoder_weights, input_shape, num_classes),
            input_shape,
            start_batch_size=config.get('batch_size', 16)
        )
    else:
        batch_size = config.get('batch_size', 8)
    
    # Build datasets
    dataset_config = {
        'rgb_path': config.get('rgb_path', 'datasets/RGB'),
        'mask_path': config.get('mask_path', 'datasets/Mask'),
        'batch_size': batch_size,
        'tile_size': tile_size,
        'seed': seed,
        'augmentations': config.get('augmentation', ['flip', 'rotate']),
        'train_val_test_split': config.get('train_val_test_split', [0.7, 0.15, 0.15]),
    }
    
    datasets = create_datasets_from_config(dataset_config)
    train_ds = datasets['train']
    val_ds = datasets['val']
    test_ds = datasets['test']
    
    # Handle deep supervision
    output_count = len(model.outputs) if isinstance(model.output, list) else 1
    if output_count > 1 and deep_supervision:
        train_data = _replicate_targets(train_ds, output_count)
        val_data = _replicate_targets(val_ds, output_count)
        steps_per_epoch = len(train_ds)
        validation_steps = len(val_ds)
    else:
        train_data = train_ds
        val_data = val_ds
        steps_per_epoch = None
        validation_steps = None
    
    # Build loss
    loss_config = {
        'loss_functions': [config.get('loss_function', 'binary_crossentropy')],
        'loss_weights': [1.0]
    }
    loss_fn = build_single_loss(loss_config)
    
    # Build optimizer
    optimizer = get_optimizer(
        config.get('optimizer', 'Adam'),
        config.get('learning_rate', 1e-3)
    )
    
    # Build metrics
    base_metrics = [
        pixel_accuracy,
        iou_score,
        dice_score,
        precision_score,
        recall_score,
        boundary_iou,
        boundary_f1,
        compactness_score,
        topological_correctness,
    ]
    metrics = base_metrics if output_count == 1 else [base_metrics] * output_count
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=metrics
    )
    
    # Resume from checkpoint if requested
    if resume:
        latest = _latest_checkpoint(checkpoint_dir)
        if latest:
            logger.info(f"Resuming from checkpoint: {latest}")
            model = tf.keras.models.load_model(latest, compile=False)
            model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
    
    # Build callbacks
    callbacks = build_callbacks(
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        run_dir=run_dir,
        early_stopping_config=config.get('early_stopping', {'enabled': True, 'patience': 10}),
        scheduler_config=config.get('scheduler', 'ReduceLROnPlateau'),
        save_best_only=True
    )
    
    # Add scheduler
    scheduler = get_scheduler(
        config.get('scheduler', 'ReduceLROnPlateau'),
        optimizer,
        config.get('epochs', 50),
        steps_per_epoch or len(train_ds),
        factor=config.get('scheduler_factor', 0.5),
        patience=config.get('scheduler_patience', 5)
    )
    if scheduler:
        callbacks.append(scheduler)
    
    # Train
    logger.info("Starting training...")
    start_time = time.time()
    
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=config.get('epochs', 50),
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=verbose
    )
    
    train_time = time.time() - start_time
    
    # Save final model
    final_model_path = os.path.join(run_dir, f"{architecture}_{encoder}_final.h5")
    model.save(final_model_path)
    
    final_metrics = {}
    serialized_history = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    for metric_name, metric_values in serialized_history.items():
        if not metric_values:
            continue
        final_metrics[f"final_{metric_name}"] = float(metric_values[-1])

    # Preserve the legacy aliases expected by older analysis scripts.
    if 'final_iou_score' in final_metrics:
        final_metrics['final_iou'] = final_metrics['final_iou_score']
    if 'final_val_iou_score' in final_metrics:
        final_metrics['final_val_iou'] = final_metrics['final_val_iou_score']
    if 'final_dice_score' in final_metrics:
        final_metrics['final_dice'] = final_metrics['final_dice_score']
    if 'final_val_dice_score' in final_metrics:
        final_metrics['final_val_dice'] = final_metrics['final_val_dice_score']

    # Create summary
    summary = {
        'experiment_id': experiment_name,
        'architecture': architecture,
        'encoder': encoder,
        'model_info': model_info,
        'config': config,
        'epochs_trained': len(history.history.get('loss', [])),
        'train_time_seconds': train_time,
        'train_time_minutes': train_time / 60,
        'final_loss': float(history.history['loss'][-1]) if history.history.get('loss') else None,
        'final_val_loss': float(history.history['val_loss'][-1]) if history.history.get('val_loss') else None,
        'history': serialized_history,
        'checkpoint_dir': checkpoint_dir,
        'final_model_path': final_model_path,
        'git_commit': get_git_commit(),
        'memory_footprint_mb': get_process_memory_mb(),
        'dataset_info': {
            'train_samples': len(train_ds.rgb_paths) if hasattr(train_ds, 'rgb_paths') else None,
            'val_samples': len(val_ds.rgb_paths) if hasattr(val_ds, 'rgb_paths') else None,
        }
    }
    summary.update(final_metrics)
    
    # Save summary
    summary_path = os.path.join(run_dir, "summary.json")
    save_json(summary_path, summary)
    
    logger.info(f"Training complete. Results saved to {run_dir}")
    logger.info(f"Final Val IoU: {summary['final_val_iou']:.4f}")
    
    return summary


def train(
    model_name: str = "Unet",
    loss_config: Dict[str, Any] = None,
    rgb_glob: str = "./datasets/RGB/*.png",
    mask_glob: str = "./datasets/Mask/*.tif",
    train_split: float = 0.8,
    input_shape: Tuple[int, int, int] = (256, 256, 3),
    batch_size: int = 4,
    epochs: int = 50,
    lr: float = 0.001,
    output_dir: str = "./results",
    experiment_name: str = None,
    deep_supervision: bool = False,
    verbose: int = 2,
    resume: bool = False,
) -> Dict[str, Any]:
    """Run standard training with a weighted sum of configured loss terms.
    
    This is the legacy training function for backward compatibility.
    For new code, use train_from_config().
    
    Parameters
    ----------
    model_name : str
        Model architecture name
    loss_config : Dict[str, Any]
        Loss configuration dictionary
    rgb_glob : str
        Glob pattern for RGB files
    mask_glob : str
        Glob pattern for mask files
    train_split : float
        Fraction for training
    input_shape : Tuple[int, int, int]
        Input shape
    batch_size : int
        Batch size
    epochs : int
        Number of epochs
    lr : float
        Learning rate
    output_dir : str
        Output directory
    experiment_name : str
        Experiment name
    deep_supervision : bool
        Whether to use deep supervision
    verbose : int
        Verbosity level
    resume : bool
        Whether to resume from checkpoint
    
    Returns
    -------
    Dict[str, Any]
        Training summary
    """
    from experiments.runner import create_experiment_folder
    
    loss_config = loss_config or {'pixel_loss': 'bce', 'pixel_weight': 1.0}
    
    run_name = experiment_name or f"standard_{model_name}"
    run_dir = create_experiment_folder(output_dir, run_name)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    log_dir = os.path.join(run_dir, "tensorboard")
    
    # Build model
    model = build_model(
        architecture=model_name,
        input_shape=input_shape,
        num_classes=1,
        deep_supervision=deep_supervision,
    )
    
    # Assemble loss functions and weights from config
    losses, weights, loss_names = build_losses(loss_config)
    
    # Import legacy dataset loading
    import glob as glob_module
    rgb_files = sorted(glob_module.glob(rgb_glob))
    mask_files = sorted(glob_module.glob(mask_glob))
    
    if len(rgb_files) != len(mask_files):
        raise ValueError("RGB and Mask file counts must match.")
    if not rgb_files:
        raise ValueError("No files found for the given RGB glob pattern.")
    
    # Split dataset
    split = int(train_split * len(rgb_files))
    train_ds = BuildingDataset(rgb_files[:split], mask_files[:split], batch_size)
    val_ds = BuildingDataset(rgb_files[split:], mask_files[split:], batch_size)
    
    # Handle deep supervision
    output_count = len(model.outputs) if isinstance(model.output, list) else 1
    if output_count > 1:
        train_data = _replicate_targets(train_ds, output_count)
        val_data = _replicate_targets(val_ds, output_count)
        steps_per_epoch = len(train_ds)
        validation_steps = len(val_ds)
    else:
        train_data = train_ds
        val_data = val_ds
        steps_per_epoch = None
        validation_steps = None
    
    # Prepare combined loss
    if len(losses) > 1:
        def combined_loss(y_true, y_pred):
            weighted_losses = [loss_fn(y_true, y_pred) * w for loss_fn, w in zip(losses, weights)]
            return tf.add_n(weighted_losses) / float(sum(weights))
        loss_to_use = combined_loss
    else:
        loss_to_use = losses[0]
    
    base_metrics = [
        pixel_accuracy,
        iou_score,
        dice_score,
        precision_score,
        recall_score,
        boundary_iou,
        boundary_f1,
        compactness_score,
        topological_correctness,
    ]
    metrics_to_use = base_metrics if output_count == 1 else [base_metrics] * output_count
    
    model.compile(
        optimizer=Adam(lr),
        loss=loss_to_use,
        metrics=metrics_to_use,
    )
    
    # Callbacks
    checkpoint_path = os.path.join(checkpoint_dir, "epoch_{epoch:03d}.h5")
    callbacks = [
        ModelCheckpoint(checkpoint_path, save_weights_only=False, save_freq="epoch"),
        CSVLogger(os.path.join(run_dir, "logs.csv"), append=resume),
        TensorBoard(log_dir=log_dir, update_freq="epoch"),
    ]
    
    # Resume if requested
    if resume:
        latest = _latest_checkpoint(checkpoint_dir)
        if latest is not None:
            model = tf.keras.models.load_model(latest, compile=False)
            model.compile(optimizer=Adam(lr), loss=loss_to_use, metrics=metrics_to_use)
    
    # Train
    start_time = time.time()
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=verbose,
    )
    train_runtime = time.time() - start_time
    
    # Summary
    summary = {
        "model_name": model_name,
        "run_name": os.path.basename(run_dir),
        "loss_config": loss_config,
        "loss_names": loss_names,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "git_commit": get_git_commit(),
        "history": history.history,
        "checkpoint_dir": checkpoint_dir,
        "train_runtime_seconds": train_runtime,
        "memory_footprint_mb": get_process_memory_mb(),
    }
    
    save_json(os.path.join(run_dir, "summary.json"), summary)
    final_model_path = os.path.join(run_dir, f"{model_name}_final.h5")
    model.save(final_model_path)
    
    return summary
