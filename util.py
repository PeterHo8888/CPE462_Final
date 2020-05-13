import tkinter as tk

def register(fn):
    global swap_fn
    swap_fn = fn

def run(box, fn):
    global swap_fn
    fn()
    swap_fn()
    box.destroy()

def create_sliders(fn, title, *argv):
    box = tk.Tk()
    box.resizable(False, False)
    tk.Label(box, text=title).pack()
    # format is (text, start, end, default)
    sliders = []
    for arg in argv:
        text, start, end, default = arg
        s = tk.Scale(box, label=text, from_=start, to=end, orient=tk.HORIZONTAL)
        s.set(default)
        s.pack()
        sliders += [s]

    tk.Button(box, text='OK', command=lambda: run(box, lambda: fn(*[val.get() for val in sliders]))).pack()
    box.mainloop()
