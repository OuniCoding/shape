"""
python modbus_monitor.py
-------------------------------------------------------------------------------
對應 PLC 暫存器表
類型	位址	說明	Python操作
Coil	M200	啟動（Str）	write_coil(0, True)
Coil	M201	停止（Stop）	write_coil(1, True)
Coil    M202    回傳PLC完成剔除訊號
Coil    M203    接收PC計數歸零訊號
Coil    M204    回傳PLC歸零完成訊號
Holding Register	D200	lo-byte, Go 數值	write_registers(0, values=[high, low])
                    D201    hi-Byte
Holding Register	D202	lo-byte, targetCount (Ct)	write_registers(2, values=[high, low])
                    D203    hi-Byte
Holding Register	D204	set 計時器設定值 (Timer Value)	write_holding_registers(204,Value)
-------------------------------------------------------------------------------
對應 Arduino 暫存器表
類型	位址	說明	Python操作
Coil	0	啟動（Str）	write_coil(0, True)
Coil	1	停止（Stop）	write_coil(1, True)
Holding Register	0	Hi-byte, Go 數值	write_registers(0, values=[high, low], device_id=slave_id)
                    1   lo-Byte
Holding Register	2	Hi-byte, targetCount (Ct)	write_registers(2, values=[high, low], device_id=slave_id)
                    3   lo-Byte
Holding Register	4	Hi-byte, triggerCount (Cs)	write_registers(4, values=[high, low], device_id=slave_id)
                    5   lo-Byte
Holding Register	6	Hi-byte, TriggerCount 回報	read_holding_registers(0, count=counter, device_id=slave_id)
                    7   lo-Byte
Holding Register	8	set 計時器設定值 (Timer Value)	read_holding_registers(8,1)
Holding Register	9	get 計時器設定值 (Timer Value)	read_holding_registers(9,1)
Holding Register	10	BufIndex 回報	read_holding_registers(6,1)
Holding Register	11	OUT Flag(en_out_flag)	read_holding_registers(7,1)
                    12~15 預留
--------------------------------------------------------------------------------
pip install pymodbus pyserial

"""
import tkinter as tk
from tkinter import ttk, messagebox
# import serial.tools.list_ports
# from pymodbus.client import ModbusSerialClient
import threading
import time
import subprocess
from pymodbus.client import ModbusTcpClient
import ipaddress
import json
import socket
from pynput import keyboard   # ⭐ 全域鍵盤

CONFIG_FILE = "config.json"

class ModbusGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Modbus TCP 上位機")
        self.root.geometry("420x520")

        self.client = None
        self.running = False

        self.counter = 0
        self.key_running = True #False

        # self.root.bind("<Key>", self.on_key_press)
        # === 全域鍵盤監聽 ===
        self.listener = keyboard.Listener(on_press=self.on_key_press)
        self.listener.start()


        self.reg_addr = 0
        self.coil_addr = 0

        # ===  連線設定 ===
        frame1 = ttk.LabelFrame(root, text="連線設定")
        frame1.pack(fill="x", padx=10, pady=5)

        style = ttk.Style()
        style.configure('Red.TButton', foreground='red')  # 設定名為 Red.TButton 的樣式，字體紅色
        style.configure('Green.TButton', foreground='green')  # 設定名為 Green.TButton 的樣式，字體綠色
        style.configure('Blue.TButton', foreground='blue')  # 設定名為 Blue.TButton 的樣式，字體藍色

        """IP set"""
        ttk.Label(frame1, text="IP:").grid(row=0, column=0, padx=5, pady=5)
        self.ip_entry = ttk.Entry(frame1, width=15)
        # self.ip_entry.insert(0, "192.168.0.5")
        self.ip_entry.grid(row=0, column=1)

        ttk.Label(frame1, text="Port:").grid(row=0, column=2)
        self.port_entry = ttk.Entry(frame1, width=6)
        # self.port_entry.insert(0, "502")
        self.port_entry.grid(row=0, column=3)

        ttk.Button(frame1, text="連線", command=self.connect_modbus, style='Green.TButton').grid(row=0, column=4)
        ttk.Button(frame1, text="掃描PLC", command=self.scan_plc, style='Blue.TButton').grid(row=0, column=5)
        # self.connect_btn = ttk.Button(frame1, text="連線", command=self.connect_modbus)
        # self.connect_btn.grid(row=0, column=4, padx=5)

        self.status_label = ttk.Label(frame1, text="未連線", foreground="red")
        # self.status_label.grid(row=0, column=5, padx=10)
        self.status_label.grid(row=1, column=0, columnspan=6)

        # === 控制區 ===
        frame2 = ttk.LabelFrame(root, text="控制命令")
        frame2.pack(fill="x", padx=10, pady=5)

        ttk.Button(frame2, text="▶ Start", command=self.cmd_start, style='Green.TButton').grid(row=0, column=0, padx=10, pady=5)
        ttk.Button(frame2, text="⏹ Stop", command=self.cmd_stop, style='Red.TButton').grid(row=0, column=1, padx=10)

        self.counter_label = ttk.Label(frame2, text="Counter: 0")
        self.counter_label.grid(row=0, column=2, padx=10)

        # === 設定區 ===
        frame3 = ttk.LabelFrame(root, text="設定命令")
        frame3.pack(fill="x", padx=10, pady=5)

        ttk.Label(frame3, text="CutTime(ms):").grid(row=0, column=0, padx=10, pady=5, sticky='e')
        self.tim_entry = ttk.Entry(frame3, width=8, justify=tk.CENTER)
        # self.tim_entry.insert(0, "100") # 初始值
        self.tim_entry.grid(row=0, column=1)
        ttk.Button(frame3, text="設定", command=self.cmd_set_tim, style='Blue.TButton').grid(row=0, column=2, padx=5)

        ttk.Label(frame3, text="暫存器起始位址:").grid(row=1, column=0, padx=10, pady=5, sticky='e')
        self.reg_entry = ttk.Entry(frame3, width=5, justify=tk.CENTER)
        # self.reg_entry.insert(0, "200") # 初始值
        self.reg_entry.grid(row=1, column=1)
        ttk.Button(frame3, text="設定", command=self.reg_set_tim, style='Blue.TButton').grid(row=1, column=2, padx=5)

        ttk.Label(frame3, text="線圈起始位址:").grid(row=2, column=0, padx=10, pady=5, sticky='e')
        self.coil_entry = ttk.Entry(frame3, width=5, justify=tk.CENTER)
        # self.coil_entry.insert(0, "200") # 初始值
        self.coil_entry.grid(row=2, column=1)
        ttk.Button(frame3, text="設定", command=self.coil_set_tim, style='Blue.TButton').grid(row=2, column=2, padx=5)

        # === 狀態顯示 ===
        frame4 = ttk.LabelFrame(root, text="暫存器資料 (每秒更新)")
        frame4.pack(fill="x", padx=10, pady=5)

        self.text = tk.Text(frame4, height=16, width=50)
        self.text.pack(padx=5, pady=5)

        # 自動刷新執行緒
        self.update_thread = threading.Thread(target=self.auto_update, daemon=True)
        self.update_thread.start()

        # 載入設定
        self.load_config()
    # ----------------------------
    def save_config(self):
        data = {
            "ip": self.ip_entry.get(),
            "port": self.port_entry.get(),
            "register": self.reg_entry.get(),
            "coil": self.coil_entry.get(),
            "cuttime": self.tim_entry.get()
        }
        with open(CONFIG_FILE, "w") as f:
            json.dump(data, f)
    def load_config(self):
        try:
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
                self.ip_entry.insert(0, data.get("ip", "192.168.0.5"))
                self.port_entry.insert(0, data.get("port", "502"))
                self.reg_entry.insert(0, data.get("register", "200"))
                self.coil_entry.insert(0, data.get("coil", "200"))
                self.tim_entry.insert(0, data.get("cuttime", "100"))
                self.reg_addr = int(self.reg_entry.get())
                self.coil_addr = int(self.coil_entry.get())
        except:
            self.ip_entry.insert(0, "192.168.0.5")
            self.port_entry.insert(0, "502")
            self.reg_entry.insert(0, "200")
            self.coil_entry.insert(0, "200")
            self.tim_entry.insert(0, "100")
            self.reg_addr = int(self.reg_entry.get())
            self.coil_addr = int(self.coil_entry.get())
        # print(self.reg_addr, self.coil_addr)
    # ----------------------------
    def scan_plc(self):
        self.text.insert(tk.END, "開始掃描...\n")

        def scan():
            base_ip = ".".join(self.ip_entry.get().split(".")[:3])

            for i in range(1, 255):
                ip = f"{base_ip}.{i}"
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(0.1)
                    result = sock.connect_ex((ip, 502))
                    if result == 0:
                        self.text.insert(tk.END, f"找到 PLC: {ip}\n")
                        self.ip_entry.delete(0, tk.END)
                        self.ip_entry.insert(0, ip)
                        sock.close()
                        return
                    sock.close()
                except:
                    pass

            self.text.insert(tk.END, "掃描完成，未找到\n")

        threading.Thread(target=scan, daemon=True).start()
    # ----------------------------
    def validate_ip_port(self, ip, port):
        # 驗證 IP
        try:
            ipaddress.ip_address(ip)
        except ValueError:
            messagebox.showerror("錯誤", f"IP 格式錯誤: {ip}")
            return False

        # 驗證 Port
        try:
            port = int(port)
            if port < 1 or port > 65535:
                raise ValueError
        except:
            messagebox.showerror("錯誤", f"Port 錯誤: {port} (需 1~65535)")
            return False

        return True

    # def on_key_press(self, event):
    #     if not self.key_running:
    #         return
    #     key = event.keysym.upper()
    #
    #     # 任意鍵 +1
    #     self.counter += 1
    #     print(f"Key: {key}, Counter: {self.counter}")
    #
    #     # 按 N → 發送
    #     if key == "N" or key == 'n':
    #         self.send_counter()

    def on_key_press(self, key):
        if not self.key_running:
            return

        # 判斷按鍵種類
        if hasattr(key, 'char') and key.char is not None:
            k = key.char.upper()  # 一般鍵
        else:
            k = str(key)  # 特殊鍵，例如 Key.up

        # Counter +1（切回主執行緒更新 UI）
        self.counter += 1
        self.root.after(0, lambda: self.counter_label.config(text=f"Counter: {self.counter}"))

        print(f"Key: {k}, Counter: {self.counter}")

        # 按 N → 傳送
        if k == 'N':
            self.send_counter()


    def ping_ip(self, ip):
        try:
            result = subprocess.run(["ping", "-n", "1", ip], stdout=subprocess.PIPE)
            return result.returncode == 0
        except:
            return False

    def connect_modbus(self):
        if self.running:
            self.running = False
            if self.client:
                self.client.close()
            self.status_label.config(text="未連線", foreground="red")
            return

        ip = self.ip_entry.get().strip()
        port_str = self.port_entry.get().strip()

        # 格式檢查
        if not self.validate_ip_port(ip, port_str):
            return

        port = int(port_str)

        if not self.ping_ip(ip):
            messagebox.showerror("錯誤", f"IP: {ip} 無法連通")
            return

        self.client = ModbusTcpClient(ip, port=port)

        if self.client.connect():
            self.status_label.config(text="已連線", foreground="green")
            self.running = True
            self.save_config()
        else:
            messagebox.showerror("錯誤", "無法連線到設備")

    # ----------------------------
    def cmd_start(self):
        if not self.client: return
        if self.running:
            self.client.write_coil(self.coil_addr, True)
            self.counter = 0
            self.key_running = True
    def cmd_stop(self):
        if not self.client: return
        if self.running:
            self.client.write_coil(self.coil_addr+1, True)
            self.key_running = False
    def send_counter(self):
        if not self.client or not self.running:
            return

        val = self.counter

        high = (val >> 16) & 0xFFFF
        low = val & 0xFFFF

        self.client.write_registers(self.reg_addr, values=[low, high])
        print(f"送出 Counter: {val}")

    def cmd_set_tim(self):
        val = int(self.tim_entry.get())
        self.save_config()
        if not self.client: return
        if self.running:
            self.client.write_register(self.reg_addr+4, val)

    def reg_set_tim(self):
        self.reg_addr = int(self.reg_entry.get())
        self.save_config()
    def coil_set_tim(self):
        self.coil_addr = int(self.coil_entry.get())
        self.save_config()

    # ----------------------------
    def auto_update(self):
        while True:
            if self.running and self.client:
                try:
                    text = []
                    # --- read holding registers ---
                    result = self.client.read_holding_registers(address=self.reg_addr, count=5)
                    if not result.isError():
                        print(result.registers)
                        for i, v in enumerate(result.registers):
                            text.append(f"D[{i}] = {v}")
                    else:
                        text.append("讀取 Register 失敗")

                    # === 讀 Coil 2~4 ===
                    coil_result = self.client.read_coils(address=self.coil_addr, count=5)
                    if not coil_result.isError():
                        coils = coil_result.bits

                        text.append("\n--- Coil 狀態 ---")
                        text.append(f"M200 = {coils[0]}")
                        text.append(f"M201 = {coils[1]}")
                        text.append(f"M202 = {coils[2]}")
                        text.append(f"M203 = {coils[3]}")
                        text.append(f"M204 = {coils[4]}")
                    else:
                        text.append("讀取 Coil 失敗")

                    self.text.delete(1.0, tk.END)
                    self.text.insert(tk.END, '\n'.join(text))
                except Exception as e:
                    self.text.delete(1.0, tk.END)
                    self.text.insert(tk.END, f"讀取錯誤: {e}")
            time.sleep(1)
# ----------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = ModbusGUI(root)
    root.mainloop()
