"""Tkinter wizard for creating and editing experiment configuration files."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from utils.config_wizard import (
    ENCODER_FILTER_PRESETS,
    GRID_PARAMETER_STATE_KEYS,
    LOSS_STRATEGIES,
    MODEL_ARCHITECTURES,
    PIXEL_LOSSES,
    SCHEDULERS,
    SEARCH_STRATEGIES,
    build_best_effort_config,
    build_file_header,
    build_full_config,
    build_hint_summary,
    build_save_payload,
    default_wizard_state,
    estimate_grid_point_count,
    load_wizard_state,
    minimize_config,
    preview_yaml,
    save_payload_to_file,
)


class ToolTip:
    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.widget = widget
        self.text = text
        self.tip_window: tk.Toplevel | None = None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, _event: tk.Event | None = None) -> None:
        if self.tip_window or not self.text:
            return
        x = self.widget.winfo_rootx() + 16
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
        self.tip_window = tk.Toplevel(self.widget)
        self.tip_window.wm_overrideredirect(True)
        self.tip_window.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            self.tip_window,
            text=self.text,
            justify="left",
            background="#fff8dc",
            relief="solid",
            borderwidth=1,
            wraplength=360,
            padx=8,
            pady=6,
        )
        label.pack()

    def hide(self, _event: tk.Event | None = None) -> None:
        if self.tip_window is not None:
            self.tip_window.destroy()
            self.tip_window = None


class ScrollableFrame(ttk.Frame):
    def __init__(self, master: tk.Misc, **kwargs) -> None:
        super().__init__(master, **kwargs)
        canvas = tk.Canvas(self, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.content = ttk.Frame(canvas)
        self.content.bind(
            "<Configure>",
            lambda event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        self._window = canvas.create_window((0, 0), window=self.content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(self._window, width=event.width))
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.content.bind_all("<MouseWheel>", lambda event: canvas.yview_scroll(int(-event.delta / 120), "units"))


class RunOutputDialog:
    """Toplevel window that streams subprocess stdout/stderr in real time."""

    _PROJECT_DIR = Path(__file__).resolve().parent.parent

    def __init__(self, parent: tk.Misc, config_path: str, mode: str) -> None:
        self._script = (
            "run_grid_search.py" if mode == "grid_search" else "run_experiment.py"
        )
        self._config_path = config_path
        self._proc: subprocess.Popen | None = None

        self.win = tk.Toplevel(parent)
        self.win.title(f"Running: {self._script}")
        self.win.geometry("900x560")
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

        top = ttk.Frame(self.win, padding=(8, 8, 8, 4))
        top.pack(fill="x")
        ttk.Label(top, text=f"Config: {config_path}", foreground="#444444").pack(side="left")

        text_frame = ttk.Frame(self.win)
        text_frame.pack(fill="both", expand=True, padx=8, pady=4)
        self._text = tk.Text(text_frame, wrap="none", state="disabled",
                             background="#1e1e1e", foreground="#d4d4d4",
                             font=("Monospace", 10))
        vsb = ttk.Scrollbar(text_frame, orient="vertical", command=self._text.yview)
        hsb = ttk.Scrollbar(text_frame, orient="horizontal", command=self._text.xview)
        self._text.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        self._text.pack(fill="both", expand=True)

        bot = ttk.Frame(self.win, padding=(8, 4, 8, 8))
        bot.pack(fill="x")
        self._status_var = tk.StringVar(value="Starting…")
        ttk.Label(bot, textvariable=self._status_var).pack(side="left")
        self._kill_btn = ttk.Button(bot, text="Stop", command=self._kill)
        self._kill_btn.pack(side="right")

        self._start()

    def _append(self, text: str) -> None:
        self._text.configure(state="normal")
        self._text.insert("end", text)
        self._text.see("end")
        self._text.configure(state="disabled")

    def _start(self) -> None:
        cmd = [sys.executable, str(self._PROJECT_DIR / self._script),
               "--config", self._config_path]
        self._status_var.set(f"Running: {' '.join(cmd)}")
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(self._PROJECT_DIR),
            bufsize=1,
        )
        threading.Thread(target=self._stream, daemon=True).start()

    def _stream(self) -> None:
        assert self._proc is not None
        try:
            for line in self._proc.stdout:  # type: ignore[union-attr]
                self.win.after(0, self._append, line)
        except ValueError:
            pass  # pipe closed
        ret = self._proc.wait()
        msg = "Finished successfully." if ret == 0 else f"Exited with code {ret}."
        self.win.after(0, self._status_var.set, msg)
        self.win.after(0, self._kill_btn.configure, {"state": "disabled"})

    def _kill(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            self._status_var.set("Process terminated.")
        self._kill_btn.configure(state="disabled")

    def _on_close(self) -> None:
        self._kill()
        self.win.destroy()


class ConfigWizardApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Experiment Config Wizard")
        self.root.geometry("1320x860")
        self.root.minsize(1100, 760)

        self.state = default_wizard_state("grid_search")
        self.current_path = str(self.state.get("save_path", ""))
        self._updating = False

        self.vars: dict[str, tk.Variable] = {}
        self.text_widgets: dict[str, tk.Text] = {}
        self.search_dimension_widgets: dict[str, tk.Widget] = {}
        self.tooltips: list[ToolTip] = []

        self._build_menu()
        self._build_layout()
        self._build_tabs()
        self._bind_state()
        self._apply_state(self.state)
        self._refresh_all()

    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="New Single Config", command=lambda: self._reset_mode("single"))
        file_menu.add_command(label="New Grid Search Config", command=lambda: self._reset_mode("grid_search"))
        file_menu.add_separator()
        file_menu.add_command(label="Load Config...", command=self._load_config)
        file_menu.add_separator()
        file_menu.add_command(label="Save", command=self._save)
        file_menu.add_command(label="Save As...", command=self._save_as)
        file_menu.add_command(label="Save & Run", command=self._save_and_run)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.destroy)
        menubar.add_cascade(label="File", menu=file_menu)

        action_menu = tk.Menu(menubar, tearoff=False)
        action_menu.add_command(label="Validate", command=self._validate)
        action_menu.add_command(label="Refresh Preview", command=self._refresh_all)
        menubar.add_cascade(label="Actions", menu=action_menu)
        self.root.configure(menu=menubar)

    def _build_layout(self) -> None:
        header = ttk.Frame(self.root, padding=(12, 10))
        header.pack(fill="x")
        header.columnconfigure(0, weight=1)
        header.columnconfigure(1, weight=1)

        config_frame = ttk.LabelFrame(header, text="Config File Actions", padding=10)
        config_frame.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        config_frame.columnconfigure(1, weight=1)
        ttk.Label(config_frame, text="Current file").grid(row=0, column=0, sticky="w")
        self.path_var = tk.StringVar(value="Unsaved")
        ttk.Label(config_frame, textvariable=self.path_var).grid(row=0, column=1, sticky="w")
        config_buttons = ttk.Frame(config_frame)
        config_buttons.grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self._make_button(config_buttons, "Load Config...", self._load_config, "Open an existing YAML config and populate the wizard from it.").pack(side="left")
        self._make_button(config_buttons, "Save", self._save, "Write the current YAML to the last saved path.").pack(side="left", padx=(8, 0))
        self._make_button(config_buttons, "Save As...", self._save_as, "Choose a new path and save the current YAML there.").pack(side="left", padx=(8, 0))
        self._make_button(config_buttons, "Auto Populate", self._auto_populate, "Fill every empty field with its built-in default value.").pack(side="left", padx=(8, 0))
        ttk.Label(
            config_frame,
            text="Load and save single combined configuration files. No inheritance is supported.",
            foreground="#666666",
            wraplength=620,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))

        body = ttk.Panedwindow(self.root, orient="horizontal")
        body.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        left = ttk.Frame(body)
        right = ttk.Frame(body, width=360)
        body.add(left, weight=4)
        body.add(right, weight=1)

        self.notebook = ttk.Notebook(left)
        self.notebook.pack(fill="both", expand=True)
        self.notebook.bind("<<NotebookTabChanged>>", lambda event: self._update_step_label())

        nav = ttk.Frame(left, padding=(0, 10, 0, 0))
        nav.pack(fill="x")
        self.step_var = tk.StringVar(value="Step 1")
        ttk.Label(nav, textvariable=self.step_var).pack(side="left")
        nav_buttons = ttk.Frame(nav)
        nav_buttons.pack(side="right")
        self._make_button(nav_buttons, "Save As", self._save_as, "Save to a new YAML path.").pack(side="right")
        self._make_button(nav_buttons, "Save & Run", self._save_and_run, "Save config and immediately launch the training process.").pack(side="right", padx=(0, 8))
        self._make_button(nav_buttons, "Validate", self._validate, "Check the current form for required fields and valid values.").pack(side="right", padx=(0, 8))
        self._make_button(nav_buttons, "Count Points", self._count_points, "Show the total number of grid-search experiment points.").pack(side="right", padx=(0, 8))
        self._make_button(nav_buttons, "Next", lambda: self._step_tab(1), "Move to the next wizard step.").pack(side="right", padx=(0, 8))
        self._make_button(nav_buttons, "Previous", lambda: self._step_tab(-1), "Move to the previous wizard step.").pack(side="right", padx=(0, 8))

        hint_frame = ttk.LabelFrame(right, text="Guidance", padding=10)
        hint_frame.pack(fill="both", expand=True)
        self.hint_text = tk.Text(hint_frame, wrap="word", height=20, state="disabled")
        self.hint_text.pack(fill="both", expand=True)

        preview_frame = ttk.LabelFrame(right, text="YAML Preview", padding=10)
        preview_frame.pack(fill="both", expand=True, pady=(10, 0))
        self.preview_text = tk.Text(preview_frame, wrap="none", state="disabled")
        self.preview_text.pack(fill="both", expand=True)

        status = ttk.Frame(self.root, padding=(12, 0, 12, 10))
        status.pack(fill="x")
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status, textvariable=self.status_var).pack(side="left")

    def _build_tabs(self) -> None:
        self._build_overview_tab()
        self._build_data_tab()
        self._build_loss_tab()
        self._build_training_tab()
        self._build_search_tab()
        self._build_review_tab()

    def _bind_state(self) -> None:
        for variable in self.vars.values():
            variable.trace_add("write", lambda *_args: self._refresh_all())

    def _new_tab(self, title: str) -> ttk.Frame:
        shell = ScrollableFrame(self.notebook)
        self.notebook.add(shell, text=title)
        shell.content.columnconfigure(0, weight=1)
        return shell.content

    def _add_section(self, parent: ttk.Frame, title: str, row: int) -> tuple[ttk.LabelFrame, int]:
        frame = ttk.LabelFrame(parent, text=title, padding=10)
        frame.grid(row=row, column=0, sticky="nsew", padx=8, pady=8)
        frame.columnconfigure(1, weight=1)
        return frame, row + 1

    def _register_var(self, key: str, variable: tk.Variable) -> None:
        self.vars[key] = variable

    def _make_button(self, parent: ttk.Frame, text: str, command, tooltip: str) -> ttk.Button:
        button = ttk.Button(parent, text=text, command=command)
        self.tooltips.append(ToolTip(button, tooltip))
        return button

    def _add_entry(
        self,
        parent: ttk.Frame,
        row: int,
        key: str,
        label: str,
        hint: str,
        width: int = 36,
        kind: str = "str",
    ) -> int:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=(2, 2))
        if kind == "bool":
            variable = tk.BooleanVar()
            self._register_var(key, variable)
            ttk.Checkbutton(parent, variable=variable).grid(row=row, column=1, sticky="w")
        else:
            variable = tk.StringVar()
            self._register_var(key, variable)
            ttk.Entry(parent, textvariable=variable, width=width).grid(row=row, column=1, sticky="ew")
        ttk.Label(parent, text=hint, foreground="#666666", wraplength=600).grid(
            row=row + 1, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )
        return row + 2

    def _add_combo(
        self,
        parent: ttk.Frame,
        row: int,
        key: str,
        label: str,
        values: list[str],
        hint: str,
    ) -> int:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=(2, 2))
        variable = tk.StringVar()
        self._register_var(key, variable)
        ttk.Combobox(parent, textvariable=variable, values=values, state="readonly").grid(
            row=row, column=1, sticky="ew"
        )
        ttk.Label(parent, text=hint, foreground="#666666", wraplength=600).grid(
            row=row + 1, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )
        return row + 2

    def _add_text(
        self,
        parent: ttk.Frame,
        row: int,
        key: str,
        label: str,
        hint: str,
        height: int = 4,
    ) -> int:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="nw", padx=(0, 10), pady=(2, 2))
        widget = tk.Text(parent, height=height, wrap="word")
        widget.grid(row=row, column=1, sticky="nsew")
        widget.bind("<KeyRelease>", lambda _event: self._refresh_all())
        self.text_widgets[key] = widget
        ttk.Label(parent, text=hint, foreground="#666666", wraplength=600).grid(
            row=row + 1, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )
        return row + 2

    def _build_overview_tab(self) -> None:
        page = self._new_tab("1. Overview")
        row = 0
        section, row = self._add_section(page, "Experiment Basics", row)
        section_row = 0
        section_row = self._add_combo(
            section,
            section_row,
            "mode",
            "Experiment mode",
            ["single", "grid_search"],
            "Options: single, grid_search. Single builds one resolved training config. Grid_search adds grid_search.parameters and lets selected search dimensions override the base config per point.",
        )
        self.vars["mode"].trace_add("write", lambda *_args: self._on_mode_change())
        section_row = self._add_entry(section, section_row, "project_name", "Project name", "Used in logs and output folders.")
        section_row = self._add_entry(section, section_row, "project_seed", "Random seed", "Keeps splits and initialization reproducible.")
        self._add_entry(
            section,
            section_row,
            "project_deterministic",
            "Deterministic TensorFlow",
            "Safer for reproducibility, sometimes slightly slower on GPU.",
            kind="bool",
        )

    def _build_data_tab(self) -> None:
        page = self._new_tab("2. Data & Model")
        row = 0
        section, row = self._add_section(page, "Data", row)
        section_row = 0
        section_row = self._add_entry(section, section_row, "data_rgb_dir", "RGB directory", "Directory with input satellite tiles.", width=50)
        section_row = self._add_entry(section, section_row, "data_mask_dir", "Mask directory", "Directory with ground-truth binary masks.", width=50)
        section_row = self._add_entry(section, section_row, "data_image_size", "Image size", "Square resize target used by the data loader.")
        section_row = self._add_entry(section, section_row, "data_batch_size", "Batch size", "Increase until you hit memory limits, then step back.")
        section_row = self._add_entry(section, section_row, "data_train_ratio", "Train ratio", "Must sum to 1.0 together with validation and test ratios.")
        section_row = self._add_entry(section, section_row, "data_val_ratio", "Validation ratio", "Used for early stopping and model selection.")
        self._add_entry(section, section_row, "data_test_ratio", "Test ratio", "Held-out evaluation split.")

        section, row = self._add_section(page, "Model", row)
        section_row = 0
        section_row = self._add_combo(
            section,
            section_row,
            "model_architecture",
            "Base architecture",
            MODEL_ARCHITECTURES,
            "Options: " + ", ".join(MODEL_ARCHITECTURES) + ". This is the actual model used in single mode. In grid-search mode it is the base/default model, and any checked search-space dimension can override it per sampled point.",
        )
        presets_hint = "Use one YAML list per line. Presets: " + ", ".join(
            f"{name}={values}" for name, values in ENCODER_FILTER_PRESETS.items()
        )
        section_row = self._add_text(section, section_row, "model_encoder_filters_text", "Encoder filters", presets_hint, height=3)
        section_row = self._add_entry(section, section_row, "model_dropout_rate", "Dropout rate", "Regularization strength inside the base UNet-style models.")
        section_row = self._add_entry(section, section_row, "model_batch_norm", "Batch normalization", "Keep enabled for stabler optimization in most runs.", kind="bool")
        self._add_entry(
            section,
            section_row,
            "model_deep_supervision",
            "Deep supervision",
            "Useful for UNet++ only. Other architectures typically ignore it.",
            kind="bool",
        )

    def _build_loss_tab(self) -> None:
        page = self._new_tab("3. Losses")
        row = 0
        section, row = self._add_section(page, "Loss Composition", row)
        section_row = 0
        section_row = self._add_combo(
            section,
            section_row,
            "loss_strategy",
            "Loss strategy",
            LOSS_STRATEGIES,
            "Options: " + ", ".join(LOSS_STRATEGIES) + ". Use 'weighted' to combine pixel, boundary and shape losses.",
        )
        section_row = self._add_combo(
            section,
            section_row,
            "loss_pixel_type",
            "Pixel loss",
            PIXEL_LOSSES,
            "Options: " + ", ".join(PIXEL_LOSSES) + ". The current repo supports these built-in pixel losses.",
        )
        section_row = self._add_entry(section, section_row, "loss_pixel_weight", "Pixel weight", "Used in weighted aggregation across active losses.")
        section_row = self._add_entry(section, section_row, "loss_boundary_enabled", "Enable boundary loss", "Adds an edge-aware Hausdorff-style term.", kind="bool")
        section_row = self._add_entry(section, section_row, "loss_boundary_weight", "Boundary weight", "Higher values emphasize crisp boundaries over region overlap.")
        section_row = self._add_entry(section, section_row, "loss_boundary_distance_threshold", "Boundary distance threshold", "Defines how wide the edge band is around object boundaries.")
        section_row = self._add_entry(section, section_row, "loss_shape_enabled", "Enable shape loss", "Adds a convexity and regularity prior on predictions.", kind="bool")
        self._add_entry(section, section_row, "loss_shape_weight", "Shape weight", "Higher values push predictions toward smoother, more regular geometry.")

    def _build_training_tab(self) -> None:
        page = self._new_tab("4. Training")
        row = 0
        section, row = self._add_section(page, "Optimizer and Scheduler", row)
        section_row = 0
        section_row = self._add_entry(section, section_row, "training_epochs", "Epochs", "Upper bound before early stopping cuts training short.")
        section_row = self._add_entry(section, section_row, "training_learning_rate", "Learning rate", "Base optimizer step size. In grid search this can be swept separately.")
        section_row = self._add_combo(
            section,
            section_row,
            "training_scheduler_type",
            "Scheduler",
            SCHEDULERS,
            "Options: " + ", ".join(SCHEDULERS) + ". Cosine is the current default. Plateau reacts to validation metrics.",
        )
        section_row = self._add_entry(section, section_row, "training_scheduler_warmup_epochs", "Warmup epochs", "Useful with cosine schedules to avoid unstable early updates.")
        section_row = self._add_entry(section, section_row, "training_scheduler_min_lr", "Minimum learning rate", "Floor used by the scheduler near the end of training.")
        section_row = self._add_entry(section, section_row, "training_early_stopping_enabled", "Enable early stopping", "Stops runs that stop improving on the monitored validation metric.", kind="bool")
        self._add_entry(section, section_row, "training_early_stopping_patience", "Early stopping patience", "Number of non-improving epochs to wait before stopping.")

        section, row = self._add_section(page, "Evaluation and Export", row)
        section_row = 0
        section_row = self._add_text(section, section_row, "evaluation_metrics_text", "Evaluation metrics", "Comma-separated metric names. Example: iou, dice, precision, recall, f1, pixel_accuracy", height=4)
        section_row = self._add_text(section, section_row, "evaluation_complexity_metrics_text", "Complexity metrics", "Comma-separated metrics such as parameter_count, flops, inference_time_ms", height=3)
        section_row = self._add_text(section, section_row, "visualization_formats_text", "Figure formats", "Comma-separated export formats for reports, for example pdf, png", height=2)
        self._add_entry(section, section_row, "export_results_dir", "Results directory", "Top-level folder for checkpoints, logs, figures, and summaries.", width=50)

    def _build_search_tab(self) -> None:
        page = self._new_tab("5. Search Space")
        row = 0
        section, row = self._add_section(page, "Grid Search Controls", row)
        section_row = 0
        section_row = self._add_entry(section, section_row, "grid_enabled", "Enable grid search", "Disable this to save a single resolved config even when starting from the grid-search template.", kind="bool")
        section_row = self._add_entry(section, section_row, "grid_auto_checkpoint", "Auto checkpoint", "Persist search-state metadata between evaluated points.", kind="bool")
        section_row = self._add_entry(section, section_row, "grid_replicate_points", "Replicate points", "Train each sampled point multiple times to measure variance.")
        section_row = self._add_entry(section, section_row, "grid_intermediate_reporting", "Intermediate reporting", "Continuously refresh aggregate outputs during long experiments.", kind="bool")
        section_row = self._add_combo(
            section,
            section_row,
            "grid_selection_strategy",
            "Selection strategy",
            SEARCH_STRATEGIES,
            "Options: " + ", ".join(SEARCH_STRATEGIES) + ". grid_search evaluates every point. random and latin_hypercube subsample the full Cartesian product.",
        )
        section_row = self._add_entry(section, section_row, "grid_selection_n_points", "Selected points", "Only used by random and latin_hypercube selection strategies.")
        section_row = self._add_entry(section, section_row, "grid_selection_random_seed", "Search random seed", "Controls deterministic subsampling for random and latin_hypercube.")
        section_row = self._add_combo(section, section_row, "grid_persistence_backend", "Persistence backend", ["json", "sqlite"], "Options: json, sqlite. Use json for simplicity or sqlite for larger, more durable searches.")
        self._add_entry(section, section_row, "grid_checkpoint_interval", "Checkpoint interval", "Save grid-search state every N updates.")

        section, row = self._add_section(page, "Grid Search Parameters (Overrides defaults)", row)
        section_row = 0
        ttk.Label(
            section,
            text=(
                "HOW THIS WORKS:\n"
                "1. If a box is CHECKED below, the grid search runner will sweep over all values you type here.\n"
                "2. When training via grid search, these values will override whatever you set in the earlier tabs (Data & Model, Losses, etc.).\n"
                "3. If a box is UNCHECKED, that parameter is ignored by the search runner, and the experiment will just use your base choice from the previous tabs."
            ),
            foreground="#005599",
            justify="left",
            font=("Sans", 11, "bold"),
            wraplength=760,
        ).grid(row=section_row, column=0, columnspan=3, sticky="w", pady=(0, 16))
        section_row += 1
        section_row = self._add_search_dimension(
            section,
            section_row,
            "grid_include_model_architecture",
            "grid_model_architecture_text",
            "Architectures",
            "Comma-separated architectures. Options: " + ", ".join(MODEL_ARCHITECTURES),
            height=2,
        )
        section_row = self._add_search_dimension(
            section,
            section_row,
            "grid_include_encoder_filters",
            "grid_encoder_filters_text",
            "Encoder filter sets",
            "One YAML list per line. Presets: " + ", ".join(f"{name}={values}" for name, values in ENCODER_FILTER_PRESETS.items()),
            height=4,
        )
        section_row = self._add_search_dimension(
            section,
            section_row,
            "grid_include_pixel_loss_type",
            "grid_pixel_loss_type_text",
            "Pixel loss types",
            "Comma-separated losses. Options: " + ", ".join(PIXEL_LOSSES),
            height=2,
        )
        section_row = self._add_search_dimension(
            section,
            section_row,
            "grid_include_boundary_loss_weight",
            "grid_boundary_loss_weight_text",
            "Boundary weights",
            "Comma-separated floats, for example 0.0, 0.3, 0.5.",
            height=2,
        )
        section_row = self._add_search_dimension(
            section,
            section_row,
            "grid_include_shape_loss_weight",
            "grid_shape_loss_weight_text",
            "Shape weights",
            "Comma-separated floats, for example 0.0, 0.1, 0.2.",
            height=2,
        )
        section_row = self._add_search_dimension(
            section,
            section_row,
            "grid_include_learning_rate",
            "grid_learning_rate_text",
            "Learning rates",
            "Comma-separated floats, for example 1.0e-4, 5.0e-4, 1.0e-3.",
            height=2,
        )
        self._add_text(section, section_row, "grid_constraints_text", "Constraints", "YAML list of skip rules. This is where you remove redundant or impossible combinations.", height=6)

    def _build_review_tab(self) -> None:
        page = self._new_tab("6. Review")
        row = 0
        section, row = self._add_section(page, "Review Checklist", row)
        ttk.Label(
            section,
            text=(
                "Use Validate to confirm that required sections are present, paths are non-empty, and ratios sum to 1.0. "
                "The live YAML preview on the right always reflects what Save would write."
            ),
            wraplength=760,
        ).grid(row=0, column=0, sticky="w")

    def _collect_state(self) -> dict[str, object]:
        state = dict(self.state)
        for key, variable in self.vars.items():
            state[key] = variable.get()
        for key, widget in self.text_widgets.items():
            state[key] = widget.get("1.0", "end").strip()
        return state

    def _apply_state(self, state: dict[str, object]) -> None:
        self._updating = True
        try:
            self.state = dict(state)
            for key, variable in self.vars.items():
                if key in state:
                    variable.set(state[key])
            for key, widget in self.text_widgets.items():
                previous_state = str(widget.cget("state"))
                if previous_state == "disabled":
                    widget.configure(state="normal")
                widget.delete("1.0", "end")
                widget.insert("1.0", str(state.get(key, "")))
                if previous_state == "disabled":
                    widget.configure(state="disabled")
            self.current_path = str(state.get("save_path", "") or self.current_path)
        finally:
            self._updating = False
        self._sync_header()

    def _sync_header(self) -> None:
        self.path_var.set(self.current_path or "Unsaved")

    def _refresh_all(self) -> None:
        if self._updating:
            return
        state = self._collect_state()
        self.state = state
        self._sync_header()
        self._sync_search_dimension_states(state)
        self._update_step_label()
        self._update_hints(state)
        self._update_preview(state)
        if bool(state.get("grid_enabled", False)):
            self.status_var.set(f"Estimated grid points before constraints: {estimate_grid_point_count(state)}")
        else:
            self.status_var.set("Single resolved config")

    def _update_hints(self, state: dict[str, object]) -> None:
        text = build_hint_summary(state)
        self.hint_text.configure(state="normal")
        self.hint_text.delete("1.0", "end")
        self.hint_text.insert("1.0", text)
        self.hint_text.configure(state="disabled")

    def _update_preview(self, state: dict[str, object]) -> None:
        text = preview_yaml(state)
        self.preview_text.configure(state="normal")
        self.preview_text.delete("1.0", "end")
        self.preview_text.insert("1.0", text)
        self.preview_text.configure(state="disabled")

    def _update_step_label(self) -> None:
        index = self.notebook.index(self.notebook.select()) + 1
        total = len(self.notebook.tabs())
        self.step_var.set(f"Step {index} of {total}")

    def _step_tab(self, delta: int) -> None:
        index = self.notebook.index(self.notebook.select())
        next_index = max(0, min(len(self.notebook.tabs()) - 1, index + delta))
        self.notebook.select(next_index)

    def _reset_mode(self, mode: str) -> None:
        state = default_wizard_state(mode)
        self.current_path = ""
        self._apply_state(state)
        self._refresh_all()

    def _on_mode_change(self) -> None:
        if self._updating:
            return
        chosen = str(self.vars["mode"].get())
        if chosen == str(self.state.get("mode", "grid_search")):
            return
        if not messagebox.askyesno(
            "Switch mode",
            "Switching mode resets the form to the template for that mode. Continue?",
        ):
            self._updating = True
            self.vars["mode"].set(str(self.state.get("mode", "grid_search")))
            self._updating = False
            return
        self._reset_mode(chosen)

    def _load_config(self) -> None:
        path = filedialog.askopenfilename(
            title="Load config",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            state = load_wizard_state(path)
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))
            return
        self.current_path = path
        state["save_path"] = path
        self._apply_state(state)
        self._refresh_all()

    def _validate(self) -> None:
        try:
            build_full_config(self._collect_state())
        except Exception as exc:
            messagebox.showerror("Validation failed", str(exc))
            return
        messagebox.showinfo("Validation", "Configuration is valid.")

    def _save(self) -> None:
        if not self.current_path:
            self._save_as()
            return
        self._save_to_path(self.current_path)

    def _save_as(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save config",
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
            initialfile="experiment.yaml",
        )
        if not path:
            return
        self._save_to_path(path)

    def _save_to_path(self, path: str) -> None:
        state = self._collect_state()
        payload, error_comment = build_best_effort_config(state)
        mode = "grid_search" if bool(state.get("grid_enabled")) else "single"
        minimized = minimize_config(payload, mode)
        header = build_file_header(payload)
        combined_comment = header + (error_comment or "")
        try:
            save_payload_to_file(minimized, path, error_comment=combined_comment)
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        self.current_path = path
        if error_comment:
            self.status_var.set(f"Saved with validation issues to {path}")
        else:
            self.status_var.set(f"Saved config to {path}")
        self._refresh_all()

    def _save_and_run(self) -> None:
        """Save config to a file then launch the training process in a streaming dialog."""
        # Ensure we have a saved path — use Save As if needed
        if not self.current_path:
            path = filedialog.asksaveasfilename(
                title="Save config before running",
                defaultextension=".yaml",
                filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
                initialfile="experiment.yaml",
            )
            if not path:
                return
            self._save_to_path(path)
        else:
            self._save_to_path(self.current_path)

        # Check for validation errors before running
        state = self._collect_state()
        _, error_comment = build_best_effort_config(state)
        if error_comment:
            if not messagebox.askyesno(
                "Validation Issues",
                "The config has validation issues:\n\n"
                + error_comment
                + "\n\nRun anyway?",
            ):
                return

        mode = "grid_search" if bool(state.get("grid_enabled")) else "single"
        RunOutputDialog(self.root, self.current_path, mode)

    def _auto_populate(self) -> None:
        """Fill every empty field with the template default for the current mode."""
        mode = str(self.vars.get("mode", tk.StringVar()).get() or "single")
        try:
            defaults = default_wizard_state(mode)
        except Exception:
            return
        current = self._collect_state()
        merged = dict(defaults)
        for k, v in current.items():
            if isinstance(v, bool):
                merged[k] = v               # booleans have no "empty" state
            elif str(v).strip():            # non-empty string → keep user value
                merged[k] = v
            # else empty string → leave the default from merged
        self._apply_state(merged)
        self._refresh_all()
        self.status_var.set("Empty fields filled with default values.")

    def _count_points(self) -> None:
        """Pop up a summary of the grid-search experiment point count."""
        state = self._collect_state()
        if not bool(state.get("grid_enabled", False)):
            messagebox.showinfo(
                "Experiment Points",
                "Grid search is not enabled.\n\n"
                "Set Mode = grid_search and enable at least one search dimension.",
            )
            return
        count = estimate_grid_point_count(state)
        active = [
            name
            for name, include_key in GRID_PARAMETER_STATE_KEYS.items()
            if bool(state.get(include_key, False))
        ]
        strategy = str(state.get("grid_selection_strategy", "grid_search"))
        n_sample = int(state.get("grid_selection_n_points", 36))

        msg = f"Active dimensions : {', '.join(active) if active else 'none'}\n\n"
        msg += f"Cartesian product  : {count} points\n"
        if strategy in {"random", "latin_hypercube"}:
            effective = min(count, n_sample) if count > 0 else 0
            msg += f"After {strategy} sampling : {effective} points (n_points={n_sample})\n"
        messagebox.showinfo("Experiment Points", msg)

    def _add_search_dimension(
        self,
        parent: ttk.Frame,
        row: int,
        include_key: str,
        text_key: str,
        label: str,
        hint: str,
        height: int = 2,
    ) -> int:
        check = tk.BooleanVar()
        self._register_var(include_key, check)
        ttk.Checkbutton(parent, variable=check).grid(row=row, column=0, sticky="nw", padx=(0, 8), pady=(2, 2))
        ttk.Label(parent, text=label).grid(row=row, column=1, sticky="nw", padx=(0, 10), pady=(2, 2))
        widget = tk.Text(parent, height=height, wrap="word")
        widget.grid(row=row, column=2, sticky="ew")
        widget.bind("<KeyRelease>", lambda _event: self._refresh_all())
        parent.columnconfigure(2, weight=1)
        self.text_widgets[text_key] = widget
        self.search_dimension_widgets[include_key] = widget
        ttk.Label(parent, text=hint, foreground="#666666", wraplength=760).grid(
            row=row + 1, column=1, columnspan=2, sticky="w", pady=(0, 8)
        )
        return row + 2

    def _sync_search_dimension_states(self, state: dict[str, object]) -> None:
        for include_key, widget in self.search_dimension_widgets.items():
            desired_state = "normal" if bool(state.get(include_key, True)) else "disabled"
            if str(widget.cget("state")) != desired_state:
                widget.configure(state=desired_state)


def main() -> None:
    root = tk.Tk()
    
    # Increase font sizes and configure default styles for better readability
    default_font = ("Sans", 11)
    root.option_add("*Font", default_font)
    
    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")
        
    style.configure(".", font=default_font)
    style.configure("TLabelframe.Label", font=("Sans", 12, "bold"))
    style.configure("TButton", font=("Sans", 11))
    
    app = ConfigWizardApp(root)
    
    # Update all Text and Entry widgets explicitly since pure Tk widgets might ignore some style configs
    for widget in root.winfo_children():
        try:
            widget.configure(font=default_font)
        except tk.TclError:
            pass
            
    root.mainloop()


if __name__ == "__main__":
    main()