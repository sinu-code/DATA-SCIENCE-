
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import pandas as pd
from collections import deque
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time

####################################################################################################################

DIRECTIONS = ['N', 'E', 'S', 'W']

####################################################################################################################

class TrafficSimulator:
    def __init__(self,
                 sim_seconds=240,
                 cycle_seconds=20,
                 min_green=4,
                 max_green=15,
                 service_rate=1.0,
                 arrival_lambda=None,
                 seed=42):
        np.random.seed(seed)
        self.sim_seconds = sim_seconds
        self.cycle_seconds = cycle_seconds
        self.min_green = min_green
        self.max_green = max_green
        self.service_rate = service_rate
        if arrival_lambda is None:
            arrival_lambda = {'N': 0.7, 'E': 0.5, 'S': 0.9, 'W': 0.4}
        self.arrival_lambda = arrival_lambda

        self.reset()

        self.priority = {'active': False, 'dir': None, 'remaining': 0, 'type': None}

####################################################################################################################

    def reset(self):
        self.queues = {d: 0 for d in DIRECTIONS}
        self.queues_deques = {d: deque() for d in DIRECTIONS}
        self.records = []
        self.passed_vehicles = []
        self.t = 0
        self.cycle_green_times = {d: self.min_green for d in DIRECTIONS}
        self.current_schedule = []
        self.running = False
        # reset priority also
        self.priority = {'active': False, 'dir': None, 'remaining': 0, 'type': None}

####################################################################################################################
    
    def simulate_arrivals(self):
        return {d: np.random.poisson(self.arrival_lambda[d]) for d in DIRECTIONS}

####################################################################################################################
    
    def allocate_green_times(self):
        total_queue = sum(max(0, q) for q in self.queues.values()) + 1e-6
        raw_times = {d: (self.queues[d] / total_queue) * self.cycle_seconds for d in DIRECTIONS}
        green_times = {}
        for d in DIRECTIONS:
            t = max(self.min_green, min(self.max_green, raw_times[d]))
            green_times[d] = int(round(t))
        total_assigned = sum(green_times.values())
        while total_assigned != self.cycle_seconds:
            if total_assigned > self.cycle_seconds:
                cand = max(DIRECTIONS, key=lambda x: (green_times[x], self.queues[x]))
                if green_times[cand] > self.min_green:
                    green_times[cand] -= 1
                    total_assigned -= 1
                else:
                    break
            else:
                cand = max(DIRECTIONS, key=lambda x: self.queues[x])
                if green_times[cand] < self.max_green:
                    green_times[cand] += 1
                    total_assigned += 1
                else:
                    break
        return green_times

####################################################################################################################
    
    def prepare_schedule(self, green_times):
        schedule = []
        for d in DIRECTIONS:
            schedule += [d] * green_times[d]
        if not schedule:
            for d in DIRECTIONS:
                schedule += [d] * self.min_green
        return schedule

    def trigger_priority(self, direction, duration_seconds=15, vehicle_type="ambulance"):
        """
        Trigger priority override for `direction` for given duration.
        vehicle_type: 'ambulance' or 'vip'
        """
        if direction not in DIRECTIONS:
            raise ValueError("Invalid direction for priority")
        self.priority = {
            'active': True,
            'dir': direction,
            'remaining': int(max(1, duration_seconds)),
            'type': vehicle_type
        }

####################################################################################################################
    
    def step_one_second(self):
        """Advance simulator by one second. Returns a dict snapshot for UI update."""
        arrivals = self.simulate_arrivals()
        for d in DIRECTIONS:
            for _ in range(arrivals[d]):
                self.queues[d] += 1
                self.queues_deques[d].append(self.t)

        if self.priority.get('active', False) and self.priority.get('remaining', 0) > 0:
            current_dir = self.priority['dir']
            vtype = self.priority['type']

            # Priority vehicles served faster
            multiplier = 4.0 if vtype == "vip" else 3.0
            effective_service = self.service_rate * multiplier

            served = min(int(np.floor(effective_service)), self.queues[current_dir])
            for _ in range(served):
                if self.queues_deques[current_dir]:
                    arrival_time = self.queues_deques[current_dir].popleft()
                    wait = self.t - arrival_time
                    self.passed_vehicles.append({
                        'dir': current_dir,
                        'pass_time': self.t,
                        'wait': wait,
                        'priority': True,
                        'type': vtype
                    })
                self.queues[current_dir] -= 1

            self.priority['remaining'] -= 1
            if self.priority['remaining'] <= 0:
                self.priority = {'active': False, 'dir': None, 'remaining': 0, 'type': None}

        else:
            if self.t % self.cycle_seconds == 0:
                self.cycle_green_times = self.allocate_green_times()
                self.current_schedule = self.prepare_schedule(self.cycle_green_times)

            if self.current_schedule:
                idx = (self.t % self.cycle_seconds) % len(self.current_schedule)
                current_dir = self.current_schedule[idx]
            else:
                current_dir = DIRECTIONS[self.t % len(DIRECTIONS)]

            served = min(int(np.floor(self.service_rate)), self.queues[current_dir])
            for _ in range(served):
                if self.queues_deques[current_dir]:
                    arrival_time = self.queues_deques[current_dir].popleft()
                    wait = self.t - arrival_time
                    self.passed_vehicles.append({'dir': current_dir, 'pass_time': self.t, 'wait': wait})
                self.queues[current_dir] -= 1

        rec = {
            'time': self.t,
            **{f'queue_{d}': self.queues[d] for d in DIRECTIONS},
            **{f'green_{d}': (1 if d == current_dir else 0) for d in DIRECTIONS}
        }
        self.records.append(rec)
        self.t += 1
        done = self.t >= self.sim_seconds
        return rec, done

####################################################################################################################
    
    def get_metrics(self):
        df_passed = pd.DataFrame(self.passed_vehicles) if self.passed_vehicles else pd.DataFrame(columns=['dir','pass_time','wait'])
        total_passed = len(df_passed)
        avg_wait = df_passed['wait'].mean() if total_passed > 0 else 0.0
        throughput_per_min = total_passed / (max(1, self.sim_seconds) / 60.0)
        return {
            'total_passed': total_passed,
            'avg_wait': avg_wait,
            'throughput_per_min': throughput_per_min,
            'final_queues': dict(self.queues)
        }

####################################################################################################################
    
    def export_results(self, path_prefix="traffic_sim"):
        df_records = pd.DataFrame(self.records)
        df_passed = pd.DataFrame(self.passed_vehicles)
        records_csv = f"{path_prefix}_records.csv"
        passed_csv = f"{path_prefix}_passed.csv"
        df_records.to_csv(records_csv, index=False)
        df_passed.to_csv(passed_csv, index=False)
        return records_csv, passed_csv


####################################################################################################################
#GUI
#####################################################################################################################

class TrafficGUI:
    def __init__(self, root):
        self.root = root
        root.title("Smart Traffic Management — Real-time Optimization")
        root.geometry("1100x700")
        self.sim = TrafficSimulator()
        self.is_running = False
        self.paused = False

        self._build_ui()
        self._init_plot()

####################################################################################################################
    
    def _build_ui(self):
        top = ttk.Frame(self.root, padding=(10,10))
        top.pack(side=tk.TOP, fill=tk.X)

        # Main control buttons
        btn_frame = ttk.Frame(top)
        btn_frame.pack(side=tk.LEFT, padx=(0,20))
        self.start_btn = ttk.Button(btn_frame, text="▶ Start", command=self.start_sim)
        self.start_btn.grid(row=0, column=0, padx=4)
        self.pause_btn = ttk.Button(btn_frame, text="⏸ Pause", command=self.pause_sim, state=tk.DISABLED)
        self.pause_btn.grid(row=0, column=1, padx=4)
        self.stop_btn = ttk.Button(btn_frame, text="⏹ Stop", command=self.stop_sim, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=2, padx=4)
        self.reset_btn = ttk.Button(btn_frame, text="↺ Reset", command=self.reset_sim)
        self.reset_btn.grid(row=0, column=3, padx=4)

        # Parameter panel
        params = ttk.LabelFrame(top, text="Simulation Parameters", padding=(10,10))
        params.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Label(params, text="Total Seconds:").grid(row=0,column=0,sticky=tk.W)
        self.entry_sim_seconds = ttk.Entry(params, width=8)
        self.entry_sim_seconds.grid(row=0,column=1,sticky=tk.W, padx=5)
        self.entry_sim_seconds.insert(0, str(self.sim.sim_seconds))

        ttk.Label(params, text="Cycle (s):").grid(row=1,column=0,sticky=tk.W)
        self.entry_cycle = ttk.Entry(params, width=8)
        self.entry_cycle.grid(row=1,column=1,sticky=tk.W, padx=5)
        self.entry_cycle.insert(0, str(self.sim.cycle_seconds))

        ttk.Label(params, text="Min Green (s):").grid(row=2,column=0,sticky=tk.W)
        self.entry_min_green = ttk.Entry(params, width=8)
        self.entry_min_green.grid(row=2,column=1,sticky=tk.W, padx=5)
        self.entry_min_green.insert(0, str(self.sim.min_green))

        ttk.Label(params, text="Max Green (s):").grid(row=3,column=0,sticky=tk.W)
        self.entry_max_green = ttk.Entry(params, width=8)
        self.entry_max_green.grid(row=3,column=1,sticky=tk.W, padx=5)
        self.entry_max_green.insert(0, str(self.sim.max_green))

        ttk.Label(params, text="Service Rate (veh/s):").grid(row=0,column=2,sticky=tk.W, padx=(20,0))
        self.entry_service = ttk.Entry(params, width=8)
        self.entry_service.grid(row=0,column=3,sticky=tk.W, padx=5)
        self.entry_service.insert(0, str(self.sim.service_rate))

        ttk.Label(params, text="Arrival λ (N,E,S,W):").grid(row=1,column=2,sticky=tk.W, padx=(20,0))
        self.entry_lambda = ttk.Entry(params, width=18)
        self.entry_lambda.grid(row=1,column=3,sticky=tk.W, padx=5)
        self.entry_lambda.insert(0, ",".join(str(self.sim.arrival_lambda[d]) for d in DIRECTIONS))

        # Metrics panel
        metrics = ttk.LabelFrame(self.root, text="Metrics", padding=(10,10))
        metrics.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=6)
        self.lbl_total = ttk.Label(metrics, text="Total Passed: 0")
        self.lbl_total.pack(anchor=tk.W, pady=2)
        self.lbl_avg_wait = ttk.Label(metrics, text="Avg Wait (s): 0.00")
        self.lbl_avg_wait.pack(anchor=tk.W, pady=2)
        self.lbl_through = ttk.Label(metrics, text="Throughput (veh/min): 0.00")
        self.lbl_through.pack(anchor=tk.W, pady=2)
        ttk.Separator(metrics, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)
        self.lbl_queues = {}
        for d in DIRECTIONS:
            l = ttk.Label(metrics, text=f"Queue {d}: 0")
            l.pack(anchor=tk.W)
            self.lbl_queues[d] = l
        ttk.Button(metrics, text="Save Results", command=self.save_results).pack(pady=(10,0))

############################################################################################################

        priority_frame = ttk.LabelFrame(self.root, text="Priority Override", padding=(8,8))
        priority_frame.pack(side=tk.RIGHT, fill=tk.X, padx=10, pady=6)
        ttk.Label(priority_frame, text="Direction:").grid(row=0, column=0, sticky=tk.W)
        self.priority_dir_var = tk.StringVar(value=DIRECTIONS[0])
        self.priority_dir_menu = ttk.Combobox(priority_frame, textvariable=self.priority_dir_var, values=DIRECTIONS, width=6, state="readonly")
        self.priority_dir_menu.grid(row=0, column=1, padx=6, pady=2)
        ttk.Label(priority_frame, text="Duration (s):").grid(row=1, column=0, sticky=tk.W)
        self.priority_duration = ttk.Entry(priority_frame, width=6)
        self.priority_duration.grid(row=1, column=1, padx=6, pady=2)
        self.priority_duration.insert(0, "15")
        ttk.Button(priority_frame, text=" Ambulance", command=self.trigger_ambulance).grid(row=2, column=0, columnspan=2, pady=(6,2))
        ttk.Button(priority_frame, text=" VIP Vehicle", command=self.trigger_vip).grid(row=3, column=0, columnspan=2, pady=(2,6))

        # Graph area
        center = ttk.Frame(self.root, padding=(8,8))
        center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.fig = Figure(figsize=(8,4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Queue lengths over time")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Queue length")
        self.canvas = FigureCanvasTkAgg(self.fig, master=center)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        bottom = ttk.Frame(self.root, padding=(8,6))
        bottom.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(bottom, textvariable=self.status_var)
        self.status_label.pack(side=tk.LEFT)

    def trigger_ambulance(self):
        try:
            dir_ = self.priority_dir_var.get()
            dur = int(self.priority_duration.get())
            self.sim.trigger_priority(dir_, dur, vehicle_type="ambulance")
            self.status_var.set(f"🚑 Ambulance priority active on {dir_} for {dur}s")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def trigger_vip(self):
        try:
            dir_ = self.priority_dir_var.get()
            dur = int(self.priority_duration.get())
            self.sim.trigger_priority(dir_, dur, vehicle_type="vip")
            self.status_var.set(f"⭐ VIP vehicle priority active on {dir_} for {dur}s")
        except Exception as e:
            messagebox.showerror("Error", str(e))

####################################################################################################################
    
    
    def _init_plot(self):
        self.times = []
        self.queues_history = {d: [] for d in DIRECTIONS}
        self.lines = {}
        colors = {'N': 'red', 'E': 'green', 'S': 'blue', 'W': 'orange'}
    
        self.ax.set_facecolor("#f9f9f9")
        self.fig.patch.set_facecolor("#f0f0f0")
        self.ax.set_title("🚦 Smart Traffic Simulation — Real-time Queue Analysis", fontsize=12, fontweight="bold")
    
        for d in DIRECTIONS:
            ln, = self.ax.plot([], [], label=f'Queue {d}', color=colors[d], linewidth=2)
            self.lines[d] = ln
    
        self.ax.legend(loc='upper right', frameon=True, facecolor='white')
        self.ax.set_xlabel("Time (seconds)", fontsize=10)
        self.ax.set_ylabel("Queue Length", fontsize=10)
        self.canvas.draw()

####################################################################################################################
    
    def start_sim(self):
        try:
            sim_seconds = int(self.entry_sim_seconds.get())
            cycle_seconds = int(self.entry_cycle.get())
            min_green = int(self.entry_min_green.get())
            max_green = int(self.entry_max_green.get())
            service = float(self.entry_service.get())
            lambdas = [float(x.strip()) for x in self.entry_lambda.get().split(",")]
            if len(lambdas) != 4:
                raise ValueError("Provide 4 lambda values")
        except Exception as e:
            messagebox.showerror("Invalid parameters", f"Check parameter values.\nError: {e}")
            return

        if self.sim.t == 0:
            self.sim.sim_seconds = sim_seconds
            self.sim.cycle_seconds = cycle_seconds
            self.sim.min_green = min_green
            self.sim.max_green = max_green
            self.sim.service_rate = service
            self.sim.arrival_lambda = dict(zip(DIRECTIONS, lambdas))
        else:
            pass

        self.is_running = True
        self.paused = False
        self.start_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_var.set("Running simulation...")
        # update loop
        self._run_loop()

####################################################################################################################

    def _run_loop(self):
        if not self.is_running or self.paused:
            return
    
        rec, done = self.sim.step_one_second()
        t = rec['time']
        self.times.append(t)
    
        for d in DIRECTIONS:
            self.queues_history[d].append(rec[f'queue_{d}'])
    
        # Update labels
        metrics = self.sim.get_metrics()
        self.lbl_total.config(text=f"🚗 Total Passed: {metrics['total_passed']}")
        self.lbl_avg_wait.config(text=f"⏱ Avg Wait: {metrics['avg_wait']:.2f} s")
        self.lbl_through.config(text=f"📊 Throughput: {metrics['throughput_per_min']:.2f} veh/min")
    
        for d in DIRECTIONS:
            q_val = metrics['final_queues'][d]
            self.lbl_queues[d].config(text=f"{d} ➜ {q_val} cars waiting")
    
        # Smooth update graph
        for d in DIRECTIONS:
            self.lines[d].set_data(self.times, self.queues_history[d])
    
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw_idle()
    
        self.status_var.set(f"Running... Time: {t}s")
        if done:
            self.status_var.set(" Simulation Finished")
            self.start_btn.config(state=tk.NORMAL)
            self.pause_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.DISABLED)
            return
    
        self.root.after(100, self._run_loop)

####################################################################################################################
    
    def pause_sim(self):
        if not self.is_running:
            return
        self.paused = not self.paused
        if self.paused:
            self.status_var.set("Paused")
            self.pause_btn.config(text="▶ Resume")
        else:
            self.status_var.set("Running...")
            self.pause_btn.config(text="⏸ Pause")
            # resume loop
            self._run_loop()

####################################################################################################################
    
    def stop_sim(self):
        if not self.is_running:
            return
        confirm = messagebox.askyesno("Stop simulation", "Are you sure you want to stop the simulation?")
        if not confirm:
            return
        self.is_running = False
        self.paused = False
        self.start_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED, text="⏸ Pause")
        self.stop_btn.config(state=tk.DISABLED)
        self.status_var.set("Stopped by user.")

####################################################################################################################
    
    def reset_sim(self):
        confirm = messagebox.askyesno("Reset", "Reset simulator and clear data?")
        if not confirm:
            return
        self.sim.reset()
        self.times = []
        self.queues_history = {d:[] for d in DIRECTIONS}
        for d in DIRECTIONS:
            self.lines[d].set_data([],[])
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw_idle()
        self.lbl_total.config(text="Total Passed: 0")
        self.lbl_avg_wait.config(text="Avg Wait (s): 0.00")
        self.lbl_through.config(text="Throughput (veh/min): 0.00")
        for d in DIRECTIONS:
            self.lbl_queues[d].config(text=f"Queue {d}: 0")
        self.status_var.set("Reset - ready")

####################################################################################################################

    def save_results(self):
        if not self.sim.records:
            messagebox.showinfo("No data", "No simulation data to save. Run simulation first.")
            return
        file = filedialog.asksaveasfilename(defaultextension=".zip",
                                            filetypes=[("CSV+passed zip","*.zip"),("All files","*.*")],
                                            title="Save results as ZIP (records + passed)")
        if not file:
            return

        import zipfile, io, os
        rec_df = pd.DataFrame(self.sim.records)
        passed_df = pd.DataFrame(self.sim.passed_vehicles)
        try:
            with zipfile.ZipFile(file, 'w') as z:
                buf1 = rec_df.to_csv(index=False).encode('utf-8')
                buf2 = passed_df.to_csv(index=False).encode('utf-8')
                z.writestr("records.csv", buf1)
                z.writestr("passed.csv", buf2)
            messagebox.showinfo("Saved", f"Results saved to: {file}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))

####################################################################################################################

def main():
    root = tk.Tk()
    style = ttk.Style(root)
    try:
        style.theme_use('clam')
    except:
        pass
    app = TrafficGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
