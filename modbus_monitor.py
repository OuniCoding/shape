"""
python modbus_monitor.py
-------------------------------------------------------------------------------
對應 Arduino 暫存器表
類型	位址	說明	Python操作
Coil	0	啟動（Str）	write_coil(0, True)
Coil	1	停止（Stop）	write_coil(1, True)
Holding Register	0	Go 數值	write_register(2, val)
Holding Register	1	targetCount (Ct)	write_register(3, val)
Holding Register	2	triggerCount (Cs)	write_register(4, val)
Holding Register	5	TriggerCount 回報	read_holding_registers(5,1)
Holding Register	6	BufIndex 回報	read_holding_registers(6,1)
Holding Register	7	OUT Flag(en_out_flag)	read_holding_registers(7,1)
Holding Register	8	set 計時器設定值 (Timer Value)	read_holding_registers(8,1)
Holding Register	9	get 計時器設定值 (Timer Value)	read_holding_registers(9,1)
--------------------------------------------------------------------------------
pip install pymodbus pyserial

"""
import tkinter as tk
from tkinter import ttk, messagebox
import serial.tools.list_ports
from pymodbus.client import ModbusSerialClient
import threading
import time

class ModbusGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Modbus RTU 上位機 (RS232)")
        self.root.geometry("420x360")

        self.client = None
        self.running = False
        self.slave_id = 1

        # === COM 選擇區 ===
        frame1 = ttk.LabelFrame(root, text="連線設定")
        frame1.pack(fill="x", padx=10, pady=5)

        ttk.Label(frame1, text="通訊埠:").grid(row=0, column=0, padx=5, pady=5)
        self.combobox = ttk.Combobox(frame1, width=10, values=self.get_com_ports())
        self.combobox.grid(row=0, column=1)
        self.combobox.set(self.get_com_ports()[0] if self.get_com_ports() else "")

        ttk.Label(frame1, text="鮑率:").grid(row=0, column=2)
        self.baud_entry = ttk.Entry(frame1, width=8)
        self.baud_entry.insert(0, "115200")
        self.baud_entry.grid(row=0, column=3)

        self.connect_btn = ttk.Button(frame1, text="連線", command=self.connect_modbus)
        self.connect_btn.grid(row=0, column=4, padx=5)

        self.status_label = ttk.Label(frame1, text="未連線", foreground="red")
        self.status_label.grid(row=0, column=5, padx=10)

        # === 控制區 ===
        frame2 = ttk.LabelFrame(root, text="控制命令")
        frame2.pack(fill="x", padx=10, pady=5)

        ttk.Button(frame2, text="▶ Start", command=self.cmd_start).grid(row=0, column=0, padx=10, pady=5)
        ttk.Button(frame2, text="⏹ Stop", command=self.cmd_stop).grid(row=0, column=1, padx=10)
        ttk.Label(frame2, text="Go:").grid(row=0, column=2)
        self.go_entry = ttk.Entry(frame2, width=6)
        self.go_entry.insert(0, "100")
        self.go_entry.grid(row=0, column=3)
        ttk.Button(frame2, text="設定 Go", command=self.cmd_set_go).grid(row=0, column=4, padx=5)

        # === 設定區 ===
        frame3 = ttk.LabelFrame(root, text="設定命令")
        frame3.pack(fill="x", padx=10, pady=5)

        ttk.Label(frame3, text="CutTime:").grid(row=0, column=0, padx=10, pady=5)
        self.tim_entry = ttk.Entry(frame3, width=8)
        self.tim_entry.insert(0, "53036")
        self.tim_entry.grid(row=0, column=1)
        ttk.Button(frame3, text="設定", command=self.cmd_set_tim).grid(row=0, column=2, padx=5)

        # === 狀態顯示 ===
        frame4 = ttk.LabelFrame(root, text="暫存器資料 (每秒更新)")
        frame4.pack(fill="x", padx=10, pady=5)

        self.text = tk.Text(frame4, height=10, width=50)
        self.text.pack(padx=5, pady=5)

        # 自動刷新執行緒
        self.update_thread = threading.Thread(target=self.auto_update, daemon=True)
        self.update_thread.start()

    # ----------------------------
    def get_com_ports(self):
        if serial.tools.list_ports.comports() == None:
            return []
        return [port.device for port in serial.tools.list_ports.comports()]

    def connect_modbus(self):
        port = self.combobox.get()
        baud = int(self.baud_entry.get())

        self.client = ModbusSerialClient(
            # method='rtu',
            port=port,
            baudrate=baud,
            parity='N',
            stopbits=1,
            bytesize=8,
            timeout=1
        )

        if self.client.connect():
            self.status_label.config(text="已連線", foreground="green")
            self.running = True
        else:
            messagebox.showerror("錯誤", "無法連線到設備")

    # ----------------------------
    def cmd_start(self):
        if not self.client: return
        self.client.write_coil(0, True, device_id=self.slave_id)

    def cmd_stop(self):
        if not self.client: return
        self.client.write_coil(1, True, device_id=self.slave_id)

    def cmd_set_go(self):
        if not self.client: return
        val = int(self.go_entry.get())
        self.client.write_register(0, val, device_id=self.slave_id)

    def cmd_set_tim(self):
        if not self.client: return
        val = int(self.tim_entry.get())
        self.client.write_register(8, val, device_id=self.slave_id)

    # ----------------------------
    def auto_update(self):
        while True:
            if self.running and self.client:
                try:
                    result = self.client.read_holding_registers(0, count=10, device_id=self.slave_id)
                    if not result.isError():
                        text = "\n".join([f"Reg[{i}] = {v}" for i, v in enumerate(result.registers)])
                        self.text.delete(1.0, tk.END)
                        self.text.insert(tk.END, text)
                except Exception as e:
                    self.text.delete(1.0, tk.END)
                    self.text.insert(tk.END, f"讀取錯誤: {e}")
            time.sleep(0.5)

# ----------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = ModbusGUI(root)
    root.mainloop()
