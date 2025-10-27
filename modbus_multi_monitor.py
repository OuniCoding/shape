
"""
這個版本支援：
✅ 自動掃描可用 COM 埠
✅ 每個埠可設定對應的 Slave ID
✅ 並行監控多個 Modbus 裝置（每秒自動刷新）
✅ 可對每台裝置分別執行 Start / Stop / 設定 Go
✅ 顯示暫存器狀態（例如 5~9）

"""

import tkinter as tk
from tkinter import ttk, messagebox
import serial.tools.list_ports
from pymodbus.client import ModbusSerialClient
import threading
import time

class ModbusDeviceFrame(ttk.LabelFrame):
    def __init__(self, parent, port, baudrate=115200, slave_id=1):
        super().__init__(parent, text=f"{port} (Slave {slave_id})")
        self.port = port
        self.baudrate = baudrate
        self.slave_id = slave_id
        self.client = None
        self.running = False

        # 狀態顯示
        self.status_label = ttk.Label(self, text="未連線", foreground="red")
        self.status_label.grid(row=0, column=0, padx=5, pady=5)

        # 控制按鈕
        ttk.Button(self, text="連線", command=self.connect).grid(row=0, column=1, padx=3)
        ttk.Button(self, text="Start", command=self.cmd_start).grid(row=0, column=2, padx=3)
        ttk.Button(self, text="Stop", command=self.cmd_stop).grid(row=0, column=3, padx=3)

        ttk.Label(self, text="Go:").grid(row=0, column=4)
        self.go_entry = ttk.Entry(self, width=5)
        self.go_entry.insert(0, "100")
        self.go_entry.grid(row=0, column=5)
        ttk.Button(self, text="設定", command=self.cmd_set_go).grid(row=0, column=6, padx=3)

        # 暫存器輸出
        self.text = tk.Text(self, height=5, width=50)
        self.text.grid(row=1, column=0, columnspan=7, padx=5, pady=5)

        # 啟動背景監控執行緒
        threading.Thread(target=self.auto_update, daemon=True).start()

    def connect(self):
        """連線到該 COM 埠的 Slave"""
        try:
            self.client = ModbusSerialClient(
                # method='rtu',
                port=self.port,
                baudrate=self.baudrate,
                parity='N',
                stopbits=1,
                bytesize=8,
                timeout=1
            )
            if self.client.connect():
                self.running = True
                self.status_label.config(text="已連線", foreground="green")
            else:
                messagebox.showerror("錯誤", f"{self.port} 連線失敗")
        except Exception as e:
            messagebox.showerror("錯誤", str(e))

    def cmd_start(self):
        if self.client:
            self.client.write_coil(0, True, device_id=self.slave_id)

    def cmd_stop(self):
        if self.client:
            self.client.write_coil(1, True, device_id=self.slave_id)

    def cmd_set_go(self):
        if self.client:
            val = int(self.go_entry.get())
            self.client.write_register(0, val, device_id=self.slave_id)

    def auto_update(self):
        while True:
            if self.running and self.client:
                try:
                    result = self.client.read_holding_registers(5, count=4, device_id=self.slave_id)
                    if not result.isError():
                        text = "\n".join([f"Reg[{i+5}] = {v}" for i, v in enumerate(result.registers)])
                        self.text.delete(1.0, tk.END)
                        self.text.insert(tk.END, text)
                except Exception as e:
                    self.text.delete(1.0, tk.END)
                    self.text.insert(tk.END, f"讀取錯誤: {e}")
            time.sleep(0.5)

class ModbusMultiMonitor:
    def __init__(self, root):
        self.root = root
        self.root.title("🧭 多埠 Modbus RTU 上位機 (RS232)")
        self.root.geometry("600x600")

        # 掃描可用埠
        self.ports = [p.device for p in serial.tools.list_ports.comports()]
        if not self.ports:
            messagebox.showerror("錯誤", "未偵測到任何 COM 埠！")
            return

        # 建立每個裝置區塊
        self.frames = []
        for i, port in enumerate(self.ports):
            slave_id = i + 1  # 預設每個裝置 ID 不同
            frame = ModbusDeviceFrame(root, port, 115200, slave_id)
            frame.pack(fill="x", padx=10, pady=5)
            self.frames.append(frame)

        # 底部說明
        ttk.Label(
            root,
            text="每秒自動刷新暫存器 [5~8]，可獨立操作各 Slave\n預設 Slave ID = 1, 2, 3... 依序遞增",
            foreground="gray"
        ).pack(pady=10)

if __name__ == "__main__":
    root = tk.Tk()
    app = ModbusMultiMonitor(root)
    root.mainloop()
