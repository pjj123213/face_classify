import csv
import json
import queue
import shutil
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from inference import DEFAULT_CHECKPOINT, FaceRaceClassifier


APP_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = APP_ROOT / DEFAULT_CHECKPOINT
FEEDBACK_DIR = APP_ROOT / "feedback"
FEEDBACK_IMAGE_DIR = FEEDBACK_DIR / "images"
FEEDBACK_LOG = FEEDBACK_DIR / "feedback_log.csv"

COLORS = {
    "bg": "#edf2f7",
    "panel": "#ffffff",
    "panel_border": "#d9e2ec",
    "ink": "#18212f",
    "muted": "#697586",
    "header": "#172033",
    "header_muted": "#b8c2d6",
    "accent": "#1f9d8a",
    "accent_dark": "#187b70",
    "canvas": "#e8eef5",
    "canvas_grid": "#d5dee9",
    "danger": "#c2410c",
}


class ProbabilityBar(ttk.Frame):
    def __init__(self, master, class_name):
        super().__init__(master, style="Panel.TFrame")
        self.class_name = class_name
        self.columnconfigure(1, weight=1)

        self.name_label = ttk.Label(self, text=class_name, width=18, style="PanelSmall.TLabel")
        self.name_label.grid(row=0, column=0, sticky="w", padx=(0, 10), pady=6)

        self.progress = ttk.Progressbar(
            self,
            orient="horizontal",
            mode="determinate",
            style="Accent.Horizontal.TProgressbar",
        )
        self.progress.grid(row=0, column=1, sticky="ew", pady=6)

        self.value_label = ttk.Label(self, text="0.00%", width=9, anchor="e", style="PanelSmall.TLabel")
        self.value_label.grid(row=0, column=2, sticky="e", padx=(10, 0), pady=6)

    def set_value(self, probability):
        self.progress["value"] = probability * 100.0
        self.value_label.configure(text=f"{probability * 100.0:.2f}%")


class FaceClassifyApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Face Classify Studio")
        self.geometry("1120x720")
        self.minsize(1020, 650)
        self.configure(background=COLORS["bg"])

        self.classifier = None
        self.current_image_path = None
        self.preview_image = None
        self.last_results = None
        self.worker_queue = queue.Queue()
        self.feedback_window = None

        self._configure_style()
        self._build_ui()
        self._set_status("正在加载模型...")
        self._start_model_loading()

    def _configure_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background=COLORS["bg"])
        style.configure(
            "Panel.TFrame",
            background=COLORS["panel"],
            relief="solid",
            borderwidth=1,
        )
        style.configure("TLabel", background=COLORS["bg"], foreground=COLORS["ink"], font=("Arial", 12))
        style.configure(
            "Panel.TLabel",
            background=COLORS["panel"],
            foreground=COLORS["ink"],
            font=("Arial", 13, "bold"),
        )
        style.configure(
            "PanelSmall.TLabel",
            background=COLORS["panel"],
            foreground=COLORS["ink"],
            font=("Arial", 11),
        )
        style.configure(
            "Title.TLabel",
            font=("Arial", 21, "bold"),
            background=COLORS["header"],
            foreground="#ffffff",
        )
        style.configure(
            "Subtitle.TLabel",
            font=("Arial", 11),
            background=COLORS["header"],
            foreground=COLORS["header_muted"],
        )
        style.configure(
            "HeaderBadge.TLabel",
            font=("Arial", 11, "bold"),
            background=COLORS["header"],
            foreground="#ffffff",
        )
        style.configure(
            "Result.TLabel",
            font=("Arial", 27, "bold"),
            background=COLORS["panel"],
            foreground=COLORS["accent_dark"],
        )
        style.configure(
            "Muted.TLabel",
            foreground=COLORS["muted"],
            background=COLORS["panel"],
            font=("Arial", 11),
        )
        style.configure(
            "Status.TLabel",
            foreground=COLORS["muted"],
            background=COLORS["bg"],
            font=("Arial", 11),
        )
        style.configure("TButton", font=("Arial", 12), padding=(14, 8))
        style.configure(
            "Accent.TButton",
            font=("Arial", 12, "bold"),
            padding=(16, 8),
            foreground="#ffffff",
            background=COLORS["accent"],
            bordercolor=COLORS["accent_dark"],
            focuscolor=COLORS["accent"],
        )
        style.map(
            "Accent.TButton",
            background=[("active", COLORS["accent_dark"]), ("disabled", "#9fb9b4")],
            foreground=[("disabled", "#edf2f7")],
        )
        style.configure(
            "Danger.TButton",
            font=("Arial", 12),
            padding=(14, 8),
            foreground="#ffffff",
            background=COLORS["danger"],
            bordercolor=COLORS["danger"],
            focuscolor=COLORS["danger"],
        )
        style.map(
            "Danger.TButton",
            background=[("active", "#9a3412"), ("disabled", "#d6a18d")],
            foreground=[("disabled", "#fff7ed")],
        )
        style.configure(
            "Accent.Horizontal.TProgressbar",
            troughcolor="#e5ebf2",
            background=COLORS["accent"],
            bordercolor="#e5ebf2",
            lightcolor=COLORS["accent"],
            darkcolor=COLORS["accent"],
            thickness=15,
        )

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        header = tk.Frame(self, background=COLORS["header"], padx=22, pady=16)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        title_box = tk.Frame(header, background=COLORS["header"])
        title_box.grid(row=0, column=0, sticky="w")
        ttk.Label(title_box, text="Face Classify Studio", style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            title_box,
            text="本地模型推理 · 7 类概率输出",
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(3, 0))

        self.device_label = ttk.Label(header, text="", anchor="e", style="HeaderBadge.TLabel")
        self.device_label.grid(row=0, column=1, sticky="e")

        main = ttk.Frame(self, padding=(20, 18, 20, 12))
        main.grid(row=1, column=0, sticky="nsew")
        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=2)
        main.rowconfigure(0, weight=1)

        left = ttk.Frame(main, style="Panel.TFrame", padding=18)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        ttk.Label(left, text="图片预览", style="Panel.TLabel").grid(row=0, column=0, sticky="w")
        self.image_canvas = tk.Canvas(
            left,
            background=COLORS["canvas"],
            highlightthickness=1,
            highlightbackground=COLORS["panel_border"],
            width=560,
            height=460,
        )
        self.image_canvas.grid(row=1, column=0, sticky="nsew", pady=(12, 10))
        self.image_canvas.bind("<Configure>", lambda _event: self._render_preview())

        self.image_path_label = ttk.Label(
            left,
            text="尚未选择图片",
            style="Muted.TLabel",
            wraplength=620,
        )
        self.image_path_label.grid(row=2, column=0, sticky="ew")

        right = ttk.Frame(main, style="Panel.TFrame", padding=20)
        right.grid(row=0, column=1, sticky="nsew", padx=(12, 0))
        right.columnconfigure(0, weight=1)

        ttk.Label(right, text="预测结果", style="Panel.TLabel").grid(row=0, column=0, sticky="w")
        self.result_label = ttk.Label(right, text="等待图片", style="Result.TLabel")
        self.result_label.grid(row=1, column=0, sticky="w", pady=(20, 4))

        self.confidence_label = ttk.Label(right, text="置信度: -", style="Muted.TLabel")
        self.confidence_label.grid(row=2, column=0, sticky="w")

        separator = ttk.Separator(right)
        separator.grid(row=3, column=0, sticky="ew", pady=20)

        self.bars_frame = ttk.Frame(right, style="Panel.TFrame")
        self.bars_frame.grid(row=4, column=0, sticky="nsew")
        self.bars_frame.columnconfigure(0, weight=1)
        self.probability_bars = {}

        footer = ttk.Frame(self, padding=(20, 10, 20, 16))
        footer.grid(row=2, column=0, sticky="ew")
        footer.columnconfigure(5, weight=1)

        self.choose_button = ttk.Button(footer, text="选择图片", command=self._choose_image)
        self.choose_button.grid(row=0, column=0, padx=(0, 10))

        self.predict_button = ttk.Button(
            footer,
            text="开始预测",
            command=self._predict_current_image,
            state="disabled",
            style="Accent.TButton",
        )
        self.predict_button.grid(row=0, column=1, padx=(0, 10))

        self.export_button = ttk.Button(
            footer,
            text="导出结果",
            command=self._export_results,
            state="disabled",
        )
        self.export_button.grid(row=0, column=2, padx=(0, 10))

        self.feedback_button = ttk.Button(
            footer,
            text="预测错误",
            command=self._open_feedback_window,
            state="disabled",
            style="Danger.TButton",
        )
        self.feedback_button.grid(row=0, column=3, padx=(0, 10))

        self.clear_button = ttk.Button(footer, text="清空", command=self._clear)
        self.clear_button.grid(row=0, column=4, padx=(0, 10))

        self.status_label = ttk.Label(footer, text="", anchor="e", style="Status.TLabel")
        self.status_label.grid(row=0, column=5, sticky="e")

    def _start_model_loading(self):
        self._set_controls_enabled(False)
        threading.Thread(target=self._load_model_worker, daemon=True).start()
        self.after(100, self._poll_worker_queue)

    def _load_model_worker(self):
        try:
            classifier = FaceRaceClassifier(CHECKPOINT_PATH)
            self.worker_queue.put(("model_loaded", classifier))
        except Exception as exc:
            self.worker_queue.put(("error", str(exc)))

    def _poll_worker_queue(self):
        try:
            while True:
                event, payload = self.worker_queue.get_nowait()
                if event == "model_loaded":
                    self.classifier = payload
                    self._build_probability_bars(payload.class_names)
                    self.device_label.configure(text=f"Device: {payload.device}")
                    self._set_status("模型加载完成")
                    self._set_controls_enabled(True)
                elif event == "prediction_done":
                    self._show_results(payload)
                    self._set_controls_enabled(True)
                    self._set_status("预测完成")
                elif event == "error":
                    self._set_controls_enabled(True)
                    self._set_status("发生错误")
                    messagebox.showerror("错误", payload)
        except queue.Empty:
            pass

        self.after(100, self._poll_worker_queue)

    def _build_probability_bars(self, class_names):
        for child in self.bars_frame.winfo_children():
            child.destroy()
        self.probability_bars = {}
        for row, class_name in enumerate(class_names):
            bar = ProbabilityBar(self.bars_frame, class_name)
            bar.grid(row=row, column=0, sticky="ew")
            self.probability_bars[class_name] = bar

    def _set_controls_enabled(self, enabled):
        state = "normal" if enabled else "disabled"
        self.choose_button.configure(state=state)
        self.clear_button.configure(state=state)

        can_predict = enabled and self.current_image_path is not None and self.classifier is not None
        self.predict_button.configure(state="normal" if can_predict else "disabled")

        can_export = enabled and self.last_results is not None
        self.export_button.configure(state="normal" if can_export else "disabled")
        self.feedback_button.configure(state="normal" if can_export else "disabled")

    def _set_status(self, text):
        self.status_label.configure(text=text)

    def _choose_image(self):
        path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        self.current_image_path = Path(path)
        self.last_results = None
        self.image_path_label.configure(text=str(self.current_image_path))
        self.result_label.configure(text="等待预测")
        self.confidence_label.configure(text="置信度: -")
        for bar in self.probability_bars.values():
            bar.set_value(0.0)
        self._render_preview()
        self._set_controls_enabled(True)
        self._set_status("图片已选择")

    def _render_preview(self):
        self.image_canvas.delete("all")
        if self.current_image_path is None:
            self.image_canvas.create_rectangle(
                24,
                24,
                max(self.image_canvas.winfo_width() - 24, 24),
                max(self.image_canvas.winfo_height() - 24, 24),
                outline=COLORS["canvas_grid"],
                dash=(6, 6),
            )
            self.image_canvas.create_text(
                self.image_canvas.winfo_width() // 2,
                self.image_canvas.winfo_height() // 2,
                text="选择一张图片开始",
                fill=COLORS["muted"],
                font=("Arial", 16),
            )
            return

        try:
            image = Image.open(self.current_image_path).convert("RGB")
        except Exception:
            return

        canvas_width = max(self.image_canvas.winfo_width(), 1)
        canvas_height = max(self.image_canvas.winfo_height(), 1)
        image.thumbnail((canvas_width - 24, canvas_height - 24), Image.Resampling.LANCZOS)
        self.preview_image = ImageTk.PhotoImage(image)
        self.image_canvas.create_image(
            canvas_width // 2,
            canvas_height // 2,
            image=self.preview_image,
            anchor="center",
        )
        self.image_canvas.create_rectangle(
            8,
            8,
            canvas_width - 9,
            canvas_height - 9,
            outline=COLORS["canvas_grid"],
        )

    def _predict_current_image(self):
        if self.classifier is None or self.current_image_path is None:
            return
        self._set_controls_enabled(False)
        self._set_status("预测中...")
        threading.Thread(target=self._predict_worker, daemon=True).start()

    def _predict_worker(self):
        try:
            results = self.classifier.predict(self.current_image_path)
            self.worker_queue.put(("prediction_done", results))
        except Exception as exc:
            self.worker_queue.put(("error", str(exc)))

    def _show_results(self, results):
        self.last_results = results
        top = results[0]
        self.result_label.configure(text=top["class_name"])
        self.confidence_label.configure(text=f"置信度: {top['probability'] * 100.0:.2f}%")

        for row in results:
            self.probability_bars[row["class_name"]].set_value(row["probability"])
        self._set_controls_enabled(True)

    def _export_results(self):
        if self.current_image_path is None or self.last_results is None:
            return

        default_name = f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        path = filedialog.asksaveasfilename(
            title="导出预测结果",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV files", "*.csv")],
        )
        if not path:
            return

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["image_path", "rank", "class_name", "probability"],
            )
            writer.writeheader()
            for rank, row in enumerate(self.last_results, start=1):
                writer.writerow(
                    {
                        "image_path": str(self.current_image_path),
                        "rank": rank,
                        "class_name": row["class_name"],
                        "probability": f"{row['probability']:.6f}",
                    }
                )

        self._set_status(f"已导出: {path}")

    def _open_feedback_window(self):
        if self.current_image_path is None or self.last_results is None or self.classifier is None:
            return
        if self.feedback_window is not None and self.feedback_window.winfo_exists():
            self.feedback_window.lift()
            return

        top = self.last_results[0]
        window = tk.Toplevel(self)
        self.feedback_window = window
        window.title("提交错误反馈")
        window.geometry("440x280")
        window.resizable(False, False)
        window.configure(background=COLORS["bg"])
        window.transient(self)

        frame = ttk.Frame(window, style="Panel.TFrame", padding=18)
        frame.pack(fill="both", expand=True, padx=14, pady=14)
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="预测错误反馈", style="Panel.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 14)
        )
        ttk.Label(frame, text="当前预测", style="PanelSmall.TLabel").grid(
            row=1, column=0, sticky="w", pady=5
        )
        ttk.Label(
            frame,
            text=f"{top['class_name']} ({top['probability'] * 100.0:.2f}%)",
            style="PanelSmall.TLabel",
        ).grid(row=1, column=1, sticky="w", pady=5)

        ttk.Label(frame, text="正确标签", style="PanelSmall.TLabel").grid(
            row=2, column=0, sticky="w", pady=5
        )
        correct_label = tk.StringVar(value=self.classifier.class_names[0])
        label_box = ttk.Combobox(
            frame,
            textvariable=correct_label,
            values=self.classifier.class_names,
            state="readonly",
        )
        label_box.grid(row=2, column=1, sticky="ew", pady=5)

        ttk.Label(frame, text="备注", style="PanelSmall.TLabel").grid(
            row=3, column=0, sticky="nw", pady=5
        )
        note_text = tk.Text(frame, height=4, width=30, wrap="word")
        note_text.grid(row=3, column=1, sticky="ew", pady=5)

        buttons = ttk.Frame(frame, style="Panel.TFrame")
        buttons.grid(row=4, column=0, columnspan=2, sticky="e", pady=(16, 0))
        ttk.Button(buttons, text="取消", command=window.destroy).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(
            buttons,
            text="保存反馈",
            style="Accent.TButton",
            command=lambda: self._save_feedback(
                correct_label.get(),
                note_text.get("1.0", "end").strip(),
                window,
            ),
        ).grid(row=0, column=1)

    def _save_feedback(self, correct_label, note, window):
        if not correct_label:
            messagebox.showwarning("缺少标签", "请选择正确标签。")
            return

        source_path = Path(self.current_image_path)
        top = self.last_results[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in source_path.stem)
        dest_dir = FEEDBACK_IMAGE_DIR / correct_label
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / f"feedback_{timestamp}_{safe_stem}{source_path.suffix.lower()}"
        shutil.copy2(source_path, dest_path)

        FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "timestamp",
            "source_image",
            "saved_image",
            "predicted_label",
            "predicted_probability",
            "correct_label",
            "all_probabilities_json",
            "note",
        ]
        write_header = not FEEDBACK_LOG.exists()
        with FEEDBACK_LOG.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "timestamp": timestamp,
                    "source_image": str(source_path),
                    "saved_image": str(dest_path),
                    "predicted_label": top["class_name"],
                    "predicted_probability": f"{top['probability']:.6f}",
                    "correct_label": correct_label,
                    "all_probabilities_json": json.dumps(self.last_results, ensure_ascii=False),
                    "note": note,
                }
            )

        self._set_status(f"已保存反馈: {correct_label}")
        window.destroy()
        messagebox.showinfo(
            "反馈已保存",
            "已记录本次错误反馈。累积一定数量后，可以运行反馈微调脚本更新模型。",
        )

    def _clear(self):
        self.current_image_path = None
        self.preview_image = None
        self.last_results = None
        self.image_path_label.configure(text="尚未选择图片")
        self.result_label.configure(text="等待图片")
        self.confidence_label.configure(text="置信度: -")
        for bar in self.probability_bars.values():
            bar.set_value(0.0)
        self._render_preview()
        self._set_controls_enabled(True)
        self._set_status("已清空")


def main():
    app = FaceClassifyApp()
    app.mainloop()


if __name__ == "__main__":
    main()
