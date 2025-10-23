"""
Interactive GUI Application for Bra-Ket Notation Education

This PyQt6 application provides an interactive interface for exploring
quantum states, operators, and the Bloch sphere visualization.

Features:
- Input quantum states interactively
- Calculate expectation values
- Visualize states on Bloch sphere
- Real-time updates
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QLineEdit, QPushButton, QGroupBox, QGridLayout,
        QTextEdit, QTabWidget, QSplitter
    )
    from PyQt6.QtCore import Qt
    PYQT_AVAILABLE = True
except ImportError:
    print("PyQt6 not available. Please install with: pip install PyQt6")
    PYQT_AVAILABLE = False

from braket_notation import (
    Ket, Bra, Operator,
    QuantumStates, PauliMatrices, BlochSphere
)


class MplCanvas(FigureCanvas):
    """Matplotlib canvas for embedding in Qt application."""
    
    def __init__(self, parent=None, width=6, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111, projection='3d')
        super().__init__(self.fig)


class BraketGUI(QMainWindow):
    """Main GUI application for bra-ket notation education."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ブラケット記法 学習ツール")
        self.setGeometry(100, 100, 1200, 800)
        
        self.current_state = QuantumStates.spin_up()
        
        self.setup_ui()
        
        self.update_calculations()
    
    def setup_ui(self):
        """Setup the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        left_panel = self.create_left_panel()
        
        right_panel = self.create_right_panel()
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        
        main_layout.addWidget(splitter)
    
    def create_left_panel(self):
        """Create the left control panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        state_group = QGroupBox("量子状態の入力")
        state_layout = QGridLayout()
        
        state_layout.addWidget(QLabel("成分1 (実部):"), 0, 0)
        self.real1_input = QLineEdit("1.0")
        state_layout.addWidget(self.real1_input, 0, 1)
        
        state_layout.addWidget(QLabel("成分1 (虚部):"), 1, 0)
        self.imag1_input = QLineEdit("0.0")
        state_layout.addWidget(self.imag1_input, 1, 1)
        
        state_layout.addWidget(QLabel("成分2 (実部):"), 2, 0)
        self.real2_input = QLineEdit("0.0")
        state_layout.addWidget(self.real2_input, 2, 1)
        
        state_layout.addWidget(QLabel("成分2 (虚部):"), 3, 0)
        self.imag2_input = QLineEdit("0.0")
        state_layout.addWidget(self.imag2_input, 3, 1)
        
        update_btn = QPushButton("状態を更新")
        update_btn.clicked.connect(self.update_state_from_input)
        state_layout.addWidget(update_btn, 4, 0, 1, 2)
        
        normalize_btn = QPushButton("正規化")
        normalize_btn.clicked.connect(self.normalize_state)
        state_layout.addWidget(normalize_btn, 5, 0, 1, 2)
        
        state_group.setLayout(state_layout)
        layout.addWidget(state_group)
        
        predef_group = QGroupBox("定義済み状態")
        predef_layout = QVBoxLayout()
        
        states = [
            ("スピン上 |↑⟩", QuantumStates.spin_up),
            ("スピン下 |↓⟩", QuantumStates.spin_down),
            ("プラス |+⟩", QuantumStates.plus_state),
            ("マイナス |-⟩", QuantumStates.minus_state),
            ("右円偏光 |R⟩", QuantumStates.right_circular),
            ("左円偏光 |L⟩", QuantumStates.left_circular),
        ]
        
        for name, state_func in states:
            btn = QPushButton(name)
            btn.clicked.connect(lambda checked, sf=state_func: self.load_predefined_state(sf))
            predef_layout.addWidget(btn)
        
        predef_group.setLayout(predef_layout)
        layout.addWidget(predef_group)
        
        results_group = QGroupBox("計算結果")
        results_layout = QVBoxLayout()
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(300)
        results_layout.addWidget(self.results_text)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        layout.addStretch()
        
        return panel
    
    def create_right_panel(self):
        """Create the right visualization panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        tabs = QTabWidget()
        
        bloch_widget = QWidget()
        bloch_layout = QVBoxLayout(bloch_widget)
        
        self.bloch_canvas = MplCanvas(self, width=6, height=6, dpi=100)
        bloch_layout.addWidget(self.bloch_canvas)
        
        refresh_btn = QPushButton("ブロッホ球を更新")
        refresh_btn.clicked.connect(self.update_bloch_sphere)
        bloch_layout.addWidget(refresh_btn)
        
        tabs.addTab(bloch_widget, "ブロッホ球")
        
        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)
        
        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_text.setHtml("""
        <h2>ブラケット記法 学習ツール</h2>
        
        <h3>使い方</h3>
        <p>1. 左側のパネルで量子状態の成分を入力します</p>
        <p>2. 「状態を更新」ボタンをクリックして状態を設定します</p>
        <p>3. 定義済み状態ボタンで標準的な状態を選択できます</p>
        <p>4. 計算結果には期待値とブロッホベクトルが表示されます</p>
        <p>5. ブロッホ球タブで状態を視覚化できます</p>
        
        <h3>量子状態の表現</h3>
        <p>量子状態 |ψ⟩ は2つの複素数で表されます：</p>
        <p>|ψ⟩ = (a + bi)|↑⟩ + (c + di)|↓⟩</p>
        
        <h3>パウリ行列</h3>
        <p>σₓ, σᵧ, σᵤ の期待値が計算されます</p>
        <p>これらはスピンの x, y, z 成分を表します</p>
        
        <h3>ブロッホ球</h3>
        <p>スピン1/2の状態は単位球面上の点として表現されます</p>
        <p>座標は (⟨σₓ⟩, ⟨σᵧ⟩, ⟨σᵤ⟩) です</p>
        """)
        info_layout.addWidget(info_text)
        
        tabs.addTab(info_widget, "使い方")
        
        layout.addWidget(tabs)
        
        return panel
    
    def update_state_from_input(self):
        """Update the quantum state from input fields."""
        try:
            real1 = float(self.real1_input.text())
            imag1 = float(self.imag1_input.text())
            real2 = float(self.real2_input.text())
            imag2 = float(self.imag2_input.text())
            
            comp1 = complex(real1, imag1)
            comp2 = complex(real2, imag2)
            
            self.current_state = Ket([comp1, comp2])
            self.update_calculations()
            
        except ValueError as e:
            self.results_text.setText(f"エラー: 数値を入力してください\n{str(e)}")
    
    def normalize_state(self):
        """Normalize the current state."""
        try:
            self.current_state = self.current_state.normalize()
            
            comp1, comp2 = self.current_state.state
            self.real1_input.setText(f"{comp1.real:.6f}")
            self.imag1_input.setText(f"{comp1.imag:.6f}")
            self.real2_input.setText(f"{comp2.real:.6f}")
            self.imag2_input.setText(f"{comp2.imag:.6f}")
            
            self.update_calculations()
            
        except Exception as e:
            self.results_text.setText(f"エラー: {str(e)}")
    
    def load_predefined_state(self, state_func):
        """Load a predefined quantum state."""
        self.current_state = state_func()
        
        comp1, comp2 = self.current_state.state
        self.real1_input.setText(f"{comp1.real:.6f}")
        self.imag1_input.setText(f"{comp1.imag:.6f}")
        self.real2_input.setText(f"{comp2.real:.6f}")
        self.imag2_input.setText(f"{comp2.imag:.6f}")
        
        self.update_calculations()
    
    def update_calculations(self):
        """Update all calculations and displays."""
        sigma_x = PauliMatrices.sigma_x()
        sigma_y = PauliMatrices.sigma_y()
        sigma_z = PauliMatrices.sigma_z()
        
        exp_x = sigma_x.expectation_value(self.current_state).real
        exp_y = sigma_y.expectation_value(self.current_state).real
        exp_z = sigma_z.expectation_value(self.current_state).real
        
        norm_squared = np.vdot(self.current_state.state, self.current_state.state).real
        
        bloch_vec = BlochSphere.state_to_bloch_vector(self.current_state) if self.current_state.is_normalized() else None
        
        results = f"""
現在の状態:
|ψ⟩ = {self.current_state.state}

正規化:
⟨ψ|ψ⟩ = {norm_squared:.6f}
正規化済み: {self.current_state.is_normalized()}

パウリ行列の期待値:
⟨σₓ⟩ = {exp_x:>8.4f}
⟨σᵧ⟩ = {exp_y:>8.4f}
⟨σᵤ⟩ = {exp_z:>8.4f}
"""
        
        if bloch_vec is not None:
            results += f"""
ブロッホベクトル:
r = ({bloch_vec[0]:.4f}, {bloch_vec[1]:.4f}, {bloch_vec[2]:.4f})
|r| = {np.linalg.norm(bloch_vec):.6f}
"""
        else:
            results += "\nブロッホベクトル: 状態を正規化してください"
        
        self.results_text.setText(results)
        
        self.update_bloch_sphere()
    
    def update_bloch_sphere(self):
        """Update the Bloch sphere visualization."""
        self.bloch_canvas.axes.clear()
        
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        
        self.bloch_canvas.axes.plot_surface(x_sphere, y_sphere, z_sphere, 
                                           alpha=0.1, color='lightblue')
        
        self.bloch_canvas.axes.plot([-1.2, 1.2], [0, 0], [0, 0], 'k-', alpha=0.3)
        self.bloch_canvas.axes.plot([0, 0], [-1.2, 1.2], [0, 0], 'k-', alpha=0.3)
        self.bloch_canvas.axes.plot([0, 0], [0, 0], [-1.2, 1.2], 'k-', alpha=0.3)
        
        self.bloch_canvas.axes.text(1.3, 0, 0, 'X', fontsize=12)
        self.bloch_canvas.axes.text(0, 1.3, 0, 'Y', fontsize=12)
        self.bloch_canvas.axes.text(0, 0, 1.3, 'Z', fontsize=12)
        
        if self.current_state.is_normalized():
            bloch_vec = BlochSphere.state_to_bloch_vector(self.current_state)
            x, y, z = bloch_vec
            
            self.bloch_canvas.axes.quiver(0, 0, 0, x, y, z, 
                                         color='red', arrow_length_ratio=0.1, linewidth=3)
            
            self.bloch_canvas.axes.scatter([x], [y], [z], color='red', s=100)
            
            self.bloch_canvas.axes.text(x*1.1, y*1.1, z*1.1, '|ψ⟩', fontsize=12)
        
        self.bloch_canvas.axes.set_xlim([-1.2, 1.2])
        self.bloch_canvas.axes.set_ylim([-1.2, 1.2])
        self.bloch_canvas.axes.set_zlim([-1.2, 1.2])
        self.bloch_canvas.axes.set_title('ブロッホ球', fontsize=14)
        
        self.bloch_canvas.draw()


def main():
    """Main entry point for the GUI application."""
    if not PYQT_AVAILABLE:
        print("PyQt6 is required to run this application.")
        print("Install with: pip install PyQt6")
        return 1
    
    app = QApplication(sys.argv)
    window = BraketGUI()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
