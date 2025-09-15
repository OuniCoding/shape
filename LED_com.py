
import os
import numpy as np
import serial
import serial.tools.list_ports
import psutil
from tkinter import Tk, Scale, Button, Label, HORIZONTAL, Entry, ttk, StringVar

brightness = np.array([0,0,0,0])
def a_data(value):
    brightness[0] = value
def b_data(value):
    brightness[1] = value
def c_data(value):
    brightness[2] = value
def d_data(value):
    brightness[3] = value

def open_com_port():
    global ser, ser_on, trigger
    com = box.get()
    if com == '':
        status.set('請選擇com port')
        sta_color.set('#00f')
        com_sta.config(fg=sta_color.get())
        return
    status.set('Connecting')
    sta_color.set('#00f')
    com_sta.config(fg=sta_color.get())
    # Label(ui, text='Connecting', background='#0f0',justify='left').grid(column=1, row=2)
    ui.update()
    try:
        ser = serial.Serial(com, 19200, 8, stopbits=1, timeout=1)
        ser_on = True

        Label(ui, text='com on', background='#0f0', justify='left').grid(column=0, row=2)
        ser.write('SA#'.encode('utf-8'))
        r_data = ser.readline()
        v = int(r_data[1:len(r_data)])
        a_slider.set(v)
        ser.write('SB#'.encode('utf-8'))
        r_data = ser.readline()
        v = int(r_data[1:len(r_data)])
        b_slider.set(v)
        ser.write('SC#'.encode('utf-8'))
        r_data = ser.readline()
        v = int(r_data[1:len(r_data)])
        c_slider.set(v)
        ser.write('SD#'.encode('utf-8'))
        r_data = ser.readline()
        v = int(r_data[1:len(r_data)])
        d_slider.set(v)
        ser.write('T#'.encode('utf-8'))
        r_data = ser.readline()
        if r_data == b'L':
            trigger = True
            # Label(ui, text='Trigger on', fg='#00f', justify='left').grid(column=0, row=13)
            trg_sta.set('Trigger on')
            fg_color.set('blue')
            sta_lab.config(fg=fg_color.get())
        elif r_data == b'H':
            trigger = False
            # Label(ui, text='Trigger off', fg='#f00', justify='left').grid(column=0, row=13)
            trg_sta.set('Trigger off')
            fg_color.set('red')
            sta_lab.config(fg=fg_color.get())
        status.set('                            ')
        # Label(ui, text='                            ', justify='left').grid(column=1, row=2)
    except: # (OSError, serial.SerialException):
        status.set('Serial Port Error!')
        sta_color.set('#f00')
        com_sta.config(fg=sta_color.get())
        # Label(ui, text='Serial Port Error!', fg='#f00', justify='left').grid(column=1, row=2)
        Label(ui, text='com off', background='#f00', justify='left').grid(column=0, row=2)
        ser_on = False
        ser.close()

def close_com_port():
    if ser_on:
        ser.close()
    Label(ui, text='com off', background='#f00', justify='left').grid(column=0, row=2)

def send_trig():
    global trigger, sta_lab
    if not ser_on:
        status.set('Serial Port is off')
        sta_color.set('#f00')
        com_sta.config(fg=sta_color.get())
        # Label(ui, text='Serial Port is off', fg='#f00', justify='left').grid(column=1, row=2)
        return
    if trigger:
        trigger = False
        # Label(ui, text='Trigger off', fg='#f00', justify='left').grid(column=0, row=13)
        status.set('                            ')
        trg_sta.set('Trigger off')
        fg_color.set('red')
        sta_lab.config(fg=fg_color.get())
        msg = 'TH#'
    else:
        trigger = True
        # Label(ui, text='Trigger on', fg='#00f', justify='left').grid(column=0, row=13)
        trg_sta.set('Trigger on')
        fg_color.set('blue')
        sta_lab.config(fg=fg_color.get())
        msg = 'TL#'
    ser.write(msg.encode('utf-8'))

    return

def data_confirm(data):
    l = 4 - len(str(data))
    data_str = ''
    if l > 0:
        for j in range(l):
            data_str = data_str + '0'
        data_str = data_str + str(data)
    return data_str
def send_data():
    if not ser_on:
        status.set('Serial Port is off')
        sta_color.set('#f00')
        com_sta.config(fg=sta_color.get())
        # Label(ui, textvariable=status, fg='#f00', justify='left').grid(column=1, row=2)
        return
    elif trigger:
        status.set('先關閉觸發')# ('First turn off the trigger!')
        sta_color.set('#f00')
        com_sta.config(fg=sta_color.get())
        return

    # status.set('                            ')
    a = a_slider.get()
    b = b_slider.get()
    c = c_slider.get()
    d = d_slider.get()
    msg = ''
    if a != brightness[0]:
        a_data(a)
        msg = data_confirm(a)
        msg = 'SA' + msg + '#'
        ser.write(msg.encode('utf-8'))
    if b != brightness[1]:
        b_data(b)
        msg = data_confirm(b)
        msg = 'SB' + msg + '#'
        ser.write(msg.encode('utf-8'))
    if c != brightness[2]:
        c_data(c)
        msg = data_confirm(c)
        msg = 'SC' + msg + '#'
        ser.write(msg.encode('utf-8'))
    if d != brightness[3]:
        d_data(d_slider.get())
        msg = data_confirm(d)
        msg = 'SD' + msg + '#'
        ser.write(msg.encode('utf-8'))
    return

def tool_quit():
    if ser_on:
        ser.close()
    ui.quit()

ser_on = False
trigger = False
process = psutil.Process(os.getpid())
p_core_ids = [0, 1, 2, 3, 4, 5, 6]
# 親和性設置 P-Core
process.cpu_affinity(p_core_ids)
print("Current CPU affinity:", process.cpu_affinity())

ui = Tk()
ui.title('光源亮度調整')
ui.geometry('250x440')

status = StringVar()
sta_color = StringVar()
trg_sta = StringVar()
fg_color = StringVar()
Label(ui, text='').grid(column=0, row=0)
Label(ui, text='選擇com port:').grid(column=0, row=1)
port_list = list(serial.tools.list_ports.comports())
coms = []
if len(port_list) > 0:
    for port in port_list:
        coms.append(port.name)
box = ttk.Combobox(ui, width=15, values=coms)
box.grid(column=1, row=1)

Label(ui, text='com off', background='#f00').grid(column=0, row=2)
status.set('')
sta_color.set('black')
com_sta = Label(ui, textvariable=status, fg=sta_color.get())
com_sta.grid(column=1, row=2)

open_button = Button(ui, text='開啟COM Port', command=open_com_port)
open_button.grid(column=1, row=3)

close_button = Button(ui, text='關閉COM Port', command=close_com_port)
close_button.grid(column=1, row=4)

Label(ui, text='').grid(column=0, row=5)
Label(ui, text='').grid(column=0, row=6)

d_slider = Scale(ui, from_=0, to=255, orient=HORIZONTAL)
d_slider.set(brightness[3])
d_slider.grid(column=1, row=7)
Label(ui, text='D通道').grid(column=0, row=7)

c_slider = Scale(ui, from_=0, to=255, orient=HORIZONTAL)
c_slider.set(brightness[3])
c_slider.grid(column=1, row=8)
Label(ui, text='C通道').grid(column=0, row=8)

b_slider = Scale(ui, from_=0, to=255, orient=HORIZONTAL)
b_slider.set(brightness[3])
b_slider.grid(column=1, row=9)
Label(ui, text='B通道').grid(column=0, row=9)

a_slider = Scale(ui, from_=0, to=255, orient=HORIZONTAL)
a_slider.set(brightness[3])
a_slider.grid(column=1, row=10)
Label(ui, text='A通道').grid(column=0, row=10)

Label(ui, text='').grid(column=0, row=11)

confirm_button = Button(ui, text='確認', command=send_data)
confirm_button.grid(column=1, row=12)

confirm_button = Button(ui, text='觸發', command=send_trig)
confirm_button.grid(column=0, row=12)
trg_sta.set('Trigger off')
fg_color.set('red')
sta_lab = Label(ui, textvariable=trg_sta, fg=fg_color.get(), justify='left')
sta_lab.grid(column=0, row=13)

exit_button = Button(ui, text='離開', command=tool_quit)
exit_button.grid(column=0, row=14)

ui.mainloop()

