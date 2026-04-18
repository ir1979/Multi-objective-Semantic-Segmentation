import csv
import glob
import os
import time
import random
import numpy as np

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard

from data.dataset import BuildingDataset, load_dataset_files, resolve_dataset_path
from experiments.runner import create_experiment_folder
from losses.loss_manager import build_losses
from models.model_factory import build_model
from training.metrics import (
    boundary_f1,
    boundary_iou,
    compactness_score,
    iou_score,
    pixel_accuracy,
    precision_score,
    recall_score,
    topological_correctness,
)
from utils.repro import dataset_hash, get_git_commit, save_json
from utils.system import get_process_memory_mb


def flatten_gradients(grads):
    """Convert a gradient list into a single flattened tensor.

    This helper is used to compute gradient directions for MGDA.
    """
    flat = []
    for g in grads:
        if g is None:
            continue
        flat.append(tf.reshape(g, [-1]))
    if not flat:
        return tf.zeros([1], dtype=tf.float32)
    return tf.concat(flat, axis=0)


def mgda_weights_two_tasks(g1, g2, eps=1e-12):
    """Compute closed-form MGDA weights for two tasks.

    This formula finds a convex combination of gradients that balances
    the two task objectives.
    """
    a = tf.reduce_sum(tf.square(g1 - g2)) + eps
    b = tf.reduce_sum((g1 - g2) * g2)
    alpha = 1.0 - b / a
    alpha = tf.clip_by_value(alpha, 0.0, 1.0)
    return tf.stack([alpha, 1.0 - alpha])


def mgda_weights_equal(K):
    """Fallback: equal weights across K tasks."""
    return tf.ones((K,), dtype=tf.float32) / float(K)


def compute_mgda_weights(grad_list):
    """Select the MGDA weight computation strategy based on number of tasks."""
    K = len(grad_list)
    if K == 1:
        return tf.ones((1,), dtype=tf.float32)
    if K == 2:
        return mgda_weights_two_tasks(grad_list[0], grad_list[1])
    return mgda_weights_equal(K)


def _latest_checkpoint(checkpoint_dir):
    """Return the newest checkpoint file in the checkpoint directory."""
    checkpoints = sorted(
        glob.glob(os.path.join(checkpoint_dir, "*.h5")),
        key=os.path.getmtime,
    )
    return checkpoints[-1] if checkpoints else None


def _replicate_targets(sequence, output_count):
    """Repeat ground truth targets for multiple model output heads."""
    while True:
        for x, y in sequence:
            yield x, tuple([y] * output_count)


def _write_csv(path, rows, headers):
    """Save a list of dictionaries to a CSV file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _build_run_name(model_name, deep_supervision):
    """Build a descriptive experiment folder name for MGDA runs."""
    name = f"mgda_{model_name}"
    if deep_supervision:
        name += "_deep"
    return name


def train_mgda(
    model_name,
    loss_config,
    rgb_glob="./datasets/RGB/*.png",
    mask_glob="./datasets/Mask/*.tif",
    train_split=0.8,
    input_shape=(256, 256, 3),
    batch_size=4,
    epochs=50,
    lr=1e-3,
    output_dir="./results",
    experiment_name=None,
    log_every=50,
    verbose=2,
    deep_supervision=False,
    resume=False,
):
    """Run MGDA-based multi-objective optimization training."""
    # Set random seeds for full reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Add timestamp to experiment name
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_name = experiment_name or _build_run_name(model_name, deep_supervision)
    run_name = f"{base_name}_{timestamp}"
    run_dir = create_experiment_folder(output_dir, run_name)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_dir, "tensorboard")
    metrics_csv = os.path.join(run_dir, "logs.csv")

    # Build model and optimizer.
    model = build_model(
        architecture=model_name,
        input_shape=input_shape,
        num_classes=1,
        deep_supervision=deep_supervision,
    )
    optimizer = Adam(lr)
    
    # Print model summary and parameter count
    print("\n📊 Model Summary:")
    model.summary()
    total_params = model.count_params()
    print(f"\n✅ Total trainable parameters: {total_params:,}")

    losses, _, loss_names = build_losses(loss_config)
    num_tasks = len(losses)
    if num_tasks == 0:
        raise ValueError("No loss functions specified in loss_config for MGDA.")

    # Load dataset file paths and verify consistency.
    rgb_glob = os.path.join(resolve_dataset_path(os.path.dirname(rgb_glob)), os.path.basename(rgb_glob))
    mask_glob = os.path.join(resolve_dataset_path(os.path.dirname(mask_glob)), os.path.basename(mask_glob))
    rgb_files, mask_files = load_dataset_files(rgb_glob, mask_glob)
    if not rgb_files:
        raise ValueError("No files found for the given RGB glob pattern.")

    if verbose >= 1:
        print("Loaded MGDA dataset files:")
        print(f"  RGB files: {len(rgb_files)}")
        print(f"  Mask files: {len(mask_files)}")
        print(f"  RGB glob: {rgb_glob}")
        print(f"  Mask glob: {mask_glob}")

    # Build dataset iterators for training and validation.
    paired_files = list(zip(rgb_files, mask_files))
    random.Random(seed).shuffle(paired_files)
    rgb_files = [rgb for rgb, _ in paired_files]
    mask_files = [mask for _, mask in paired_files]
    split = int(train_split * len(rgb_files))
    train_ds = BuildingDataset(rgb_files[:split], mask_files[:split], batch_size)
    val_ds = BuildingDataset(rgb_files[split:], mask_files[split:], batch_size)

    output_count = len(model.outputs) if isinstance(model.output, list) else 1
    if output_count > 1:
        train_seq = _replicate_targets(train_ds, output_count)
        val_seq = _replicate_targets(val_ds, output_count)
        train_steps = len(train_ds)
        val_steps = len(val_ds)
    else:
        train_seq = train_ds
        val_seq = val_ds
        train_steps = len(train_ds)
        val_steps = len(val_ds)

    if resume:
        latest_checkpoint = _latest_checkpoint(checkpoint_dir)
        if latest_checkpoint is not None:
            model = tf.keras.models.load_model(latest_checkpoint, compile=False)
            optimizer = Adam(lr)

    if verbose >= 1:
        print("MGDA training configuration:")
        print(f"  model: {model_name}")
        print(f"  deep_supervision: {deep_supervision}")
        print(f"  outputs: {output_count}")
        print(f"  batch_size: {batch_size}")
        print(f"  epochs: {epochs}")
        print(f"  learning_rate: {lr}")
        print(f"  losses: {', '.join(loss_names)}")
        print(f"  train_steps: {train_steps}")
        print(f"  val_steps: {val_steps}")
        print(f"  log_every: {log_every}")

    writer = tf.summary.create_file_writer(tensorboard_dir)
    rows = []
    total_train_start = time.time()
    total_val_time = 0.0
    
    # Best checkpoint tracking
    best_val_loss = float('inf')
    best_epoch = 0
    pareto_objectives = []  # For Pareto front plot
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        epoch_train_loss = 0.0
        epoch_train_iou = 0.0
        step = 0

        # Training loop for one epoch.
        for x_batch, y_batch in train_seq:
            step += 1
            with tf.GradientTape(persistent=True) as tape:
                y_pred = model(x_batch, training=True)
                if isinstance(y_pred, list):
                    y_preds = y_pred
                else:
                    y_preds = [y_pred]

                y_true = y_batch[0] if isinstance(y_batch, (list, tuple)) else y_batch

                # Compute each task loss separately.
                task_losses = []
                for loss_fn in losses:
                    if len(y_preds) > 1:
                        loss_value = tf.add_n([loss_fn(y_true, p) for p in y_preds]) / float(len(y_preds))
                    else:
                        loss_value = loss_fn(y_true, y_preds[0])
                    task_losses.append(loss_value)

            task_grads = [tape.gradient(loss_value, model.trainable_variables) for loss_value in task_losses]
            del tape

            # Compute MGDA weights from task gradients.
            flat_grads = [flatten_gradients(g) for g in task_grads]
            alphas = compute_mgda_weights(flat_grads)

            # Create a weighted combination of task gradients.
            combined_grads = []
            for var_idx in range(len(model.trainable_variables)):
                var_grads = [task_grads[k][var_idx] for k in range(num_tasks)]
                if all(g is None for g in var_grads):
                    combined_grads.append(None)
                    continue
                grad_stack = [g if g is not None else tf.zeros_like(model.trainable_variables[var_idx]) for g in var_grads]
                grad_stack = tf.stack(grad_stack, axis=0)
                alpha_expanded = tf.reshape(alphas, (num_tasks,) + (1,) * (len(grad_stack.shape) - 1))
                combined_grads.append(tf.reduce_sum(alpha_expanded * grad_stack, axis=0))

            optimizer.apply_gradients(zip(combined_grads, model.trainable_variables))

            batch_loss = tf.add_n(task_losses) / float(num_tasks)
            epoch_train_loss += float(batch_loss)
            epoch_train_iou += float(iou_score(y_batch, y_preds[-1]))

            if step % log_every == 0:
                print(
                    f"  Step {step}/{train_steps} | loss={epoch_train_loss / step:.4f} "
                    f"| iou={epoch_train_iou / step:.4f}"
                )
            if step >= train_steps:
                break

        train_loss_epoch = epoch_train_loss / max(train_steps, 1)
        train_iou_epoch = epoch_train_iou / max(train_steps, 1)

        # Validation loop with additional evaluation metrics.
        val_loss_epoch = 0.0
        val_iou_epoch = 0.0
        val_pixel_accuracy = 0.0
        val_boundary_iou = 0.0
        val_boundary_f1 = 0.0
        val_compactness = 0.0
        val_topological = 0.0
        val_count = 0

        val_start = time.time()
        for x_val, y_val in val_seq:
            y_pred_val = model(x_val, training=False)
            if isinstance(y_pred_val, list):
                y_preds_val = y_pred_val
            else:
                y_preds_val = [y_pred_val]

            y_val_true = y_val[0] if isinstance(y_val, (list, tuple)) else y_val
            val_task_losses = [
                tf.add_n([loss_fn(y_val_true, p) for p in y_preds_val]) / float(len(y_preds_val))
                if len(y_preds_val) > 1
                else loss_fn(y_val_true, y_preds_val[0])
                for loss_fn in losses
            ]
            pred = y_preds_val[-1]
            val_loss_epoch += float(tf.add_n(val_task_losses) / float(num_tasks))
            val_iou_epoch += float(iou_score(y_val, pred))
            val_pixel_accuracy += float(pixel_accuracy(y_val, pred))
            val_boundary_iou += float(boundary_iou(y_val, pred))
            val_boundary_f1 += float(boundary_f1(y_val, pred))
            val_compactness += float(compactness_score(y_val, pred))
            val_topological += float(topological_correctness(y_val, pred))
            val_count += 1
            if val_count >= val_steps:
                break
        val_batch_time = time.time() - val_start
        total_val_time += val_batch_time

        # Average validation metrics across the full validation set.
        val_loss_epoch /= max(val_count, 1)
        val_iou_epoch /= max(val_count, 1)
        val_pixel_accuracy /= max(val_count, 1)
        val_boundary_iou /= max(val_count, 1)
        val_boundary_f1 /= max(val_count, 1)
        val_compactness /= max(val_count, 1)
        val_topological /= max(val_count, 1)

        with writer.as_default():
            tf.summary.scalar("train_loss", train_loss_epoch, step=epoch)
            tf.summary.scalar("train_iou", train_iou_epoch, step=epoch)
            tf.summary.scalar("val_loss", val_loss_epoch, step=epoch)
            tf.summary.scalar("val_iou", val_iou_epoch, step=epoch)
            tf.summary.scalar("val_pixel_accuracy", val_pixel_accuracy, step=epoch)
            tf.summary.scalar("val_boundary_iou", val_boundary_iou, step=epoch)
            tf.summary.scalar("val_boundary_f1", val_boundary_f1, step=epoch)
            tf.summary.scalar("val_compactness", val_compactness, step=epoch)
            tf.summary.scalar("val_topological_correctness", val_topological, step=epoch)
            for idx, alpha in enumerate(alphas.numpy().tolist()):
                tf.summary.scalar(f"mgda_alpha_{idx}", alpha, step=epoch)

        print(
            f"Epoch {epoch + 1} summary: loss={train_loss_epoch:.4f}, "
            f"iou={train_iou_epoch:.4f}, val_loss={val_loss_epoch:.4f}, val_iou={val_iou_epoch:.4f}"
        )

        rows.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss_epoch,
                "train_iou": train_iou_epoch,
                "val_loss": val_loss_epoch,
                "val_iou": val_iou_epoch,
                "val_pixel_accuracy": val_pixel_accuracy,
                "val_boundary_iou": val_boundary_iou,
                "val_boundary_f1": val_boundary_f1,
                "val_compactness": val_compactness,
                "val_topological_correctness": val_topological,
                **{f"alpha_{idx}": float(alpha) for idx, alpha in enumerate(alphas.numpy().tolist())},
            }
        )

        model.save(os.path.join(checkpoint_dir, f"epoch_{epoch + 1:03d}.h5"))
        
        # Save best checkpoint based on validation loss
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            best_epoch = epoch + 1
            model.save(os.path.join(checkpoint_dir, "best_model.h5"))
            print(f"  ✅ New best model saved at epoch {best_epoch} (val_loss: {best_val_loss:.4f})")
            
            # Cleanup old checkpoints to save disk space (keep only last 3 and best)
            all_checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "epoch_*.h5")))
            if len(all_checkpoints) > 3:
                for old_checkpoint in all_checkpoints[:-3]:
                    os.remove(old_checkpoint)
                    print(f"  🗑️  Cleaned old checkpoint: {os.path.basename(old_checkpoint)}")
        
        # Collect objective values for Pareto front
        current_objectives = [val_loss_epoch, 1.0 - val_iou_epoch]
        pareto_objectives.append(current_objectives)

    _write_csv(metrics_csv, rows, headers=list(rows[0].keys()) if rows else ["epoch", "train_loss"])
    
    # Generate training curves figure
    from visualization.visualization import plot_loss_curves, save_pareto_plot
    
    # Prepare history dictionary for plotting
    history = {
        'train_loss': [row['train_loss'] for row in rows],
        'val_loss': [row['val_loss'] for row in rows],
        'train_iou': [row['train_iou'] for row in rows],
        'val_iou': [row['val_iou'] for row in rows],
    }
    plot_loss_curves(os.path.join(run_dir, "training_curves.png"), history)
    
    # Generate Pareto front plot
    if len(pareto_objectives) > 0:
        save_pareto_plot(os.path.join(run_dir, "pareto_front.png"), pareto_objectives, 
                          labels=["Validation Loss", "1 - IoU (Error]"])
    
    summary = {
        "model_name": model_name,
        "run_name": os.path.basename(run_dir),
        "loss_config": loss_config,
        "loss_names": loss_names,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "dataset_hash": dataset_hash(rgb_files + mask_files),
        "git_commit": get_git_commit(),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "final_train_loss": train_loss_epoch,
        "final_train_iou": train_iou_epoch,
        "final_val_loss": val_loss_epoch,
        "final_val_iou": val_iou_epoch,
        "final_val_pixel_accuracy": val_pixel_accuracy,
        "final_val_boundary_iou": val_boundary_iou,
        "final_val_boundary_f1": val_boundary_f1,
        "final_val_compactness": val_compactness,
        "final_val_topological_correctness": val_topological,
        "train_runtime_seconds": time.time() - total_train_start,
        "validation_runtime_seconds": total_val_time,
        "memory_footprint_mb": get_process_memory_mb(),
    }
    save_json(os.path.join(run_dir, "summary.json"), summary)
    final_model_path = os.path.join(run_dir, f"{model_name}_mgda_final.h5")
    model.save(final_model_path)
    
    # Generate sample predictions for first 100 validation images
    print("\n🖼️  Generating sample predictions for visualization...")
    from visualization.visualization import save_sample_predictions
    sample_rgb = []
    sample_gt = []
    sample_pred = []
    
    # Reset validation iterator
    val_count = 0
    for x_val, y_val in val_seq:
        y_pred_val = model(x_val, training=False)
        if isinstance(y_pred_val, list):
            pred = y_pred_val[-1]
        else:
            pred = y_pred_val
        
        # Extract first image from batch (handle both Tensor and numpy array inputs)
        def safe_numpy(x):
            return x.numpy() if hasattr(x, 'numpy') else x
        
        sample_rgb.append(safe_numpy(x_val[0]))
        sample_gt.append(safe_numpy(y_val[0]) if not isinstance(y_val, tuple) else safe_numpy(y_val[0][0]))
        sample_pred.append(safe_numpy(pred[0]))
        
        val_count += 1
        if val_count >= 100:
            break
    
    # Save visualization grid
    save_sample_predictions(
        os.path.join(run_dir, "sample_predictions.png"),
        sample_rgb[:16],  # Show first 16 samples in grid
        sample_gt[:16],
        sample_pred[:16],
        titles=["RGB Input", "Ground Truth", "Prediction"]
    )
    print(f"✅ Saved sample predictions to sample_predictions.png")
    
    print(f"\n✅ Training completed successfully!")
    print(f"  Best epoch: {best_epoch} with validation loss: {best_val_loss:.4f}")
    print(f"  Results saved to: {run_dir}")
    print(f"  Generated figures: training_curves.png, pareto_front.png")
    
    return summary
