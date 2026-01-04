"""
Advanced Digital Certificate Fraud Detection System
Unmasking the Underground: AI-Powered Real-Time Certificate Fraud Detection

FIXED VERSION WITH:
- Improved fraud detection sensitivity
- Better risk calculation
- More realistic fraud triggers
- Visible fraud patterns
- Adjustable detection thresholds

Author: Sarthak Milind Upasani
University of Hertfordshire - MSc Cyber Security
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import time
import random
import datetime
import json
import os
from collections import defaultdict

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from cryptography import x509
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import serialization
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


class AdvancedSplashScreen:
    """Professional animated splash screen"""

    def __init__(self, parent):
        self.parent = parent
        self.splash = tk.Toplevel(parent)
        self.splash.title("Loading...")
        self.splash.geometry("700x500")
        self.splash.configure(bg='#0a0a1a')
        self.splash.overrideredirect(True)
        self.splash.attributes('-topmost', True)

        x = (self.splash.winfo_screenwidth() // 2) - 350
        y = (self.splash.winfo_screenheight() // 2) - 250
        self.splash.geometry(f"+{x}+{y}")

        main = tk.Label(self.splash, bg='#0a0a1a', height=500)
        main.pack(fill=tk.BOTH, expand=True)

        title = tk.Label(main, text="🔐", font=("Arial", 60), fg="#00ffff", bg='#0a0a1a')
        title.pack(pady=20)

        title_text = tk.Label(main, text="Certificate Fraud Detection",
                        font=("Arial", 28, "bold"), fg="#00ffff", bg='#0a0a1a')
        title_text.pack()

        subtitle = tk.Label(main, text="AI-Powered Real-Time Monitoring System",
                           font=("Arial", 14), fg="#00ff88", bg='#0a0a1a')
        subtitle.pack(pady=10)

        progress_frame = tk.Frame(main, bg='#1a1a2e', height=8)
        progress_frame.pack(pady=30, padx=50, fill=tk.X)

        self.progress_fill = tk.Label(progress_frame, bg='#00ffff', height=1)
        self.progress_fill.pack(side=tk.LEFT, fill=tk.Y)

        self.status = tk.Label(main, text="Initializing system...",
                              font=("Arial", 11), fg="#ffffff", bg='#0a0a1a')
        self.status.pack(pady=15)

        self.dots_frame = tk.Label(main, font=("Arial", 14), fg="#00ff88", bg='#0a0a1a')
        self.dots_frame.pack()

        self.dot_count = 0
        self.progress_value = 0
        self.animate()

    def animate(self):
        if self.progress_value < 100:
            self.progress_value += random.randint(2, 5)
            if self.progress_value > 100:
                self.progress_value = 100

            width = int((self.progress_value / 100) * 600)
            self.progress_fill.config(width=width)

            self.dot_count = (self.dot_count + 1) % 4
            dots = "." * self.dot_count
            self.status.config(text=f"Loading... {self.progress_value}% {dots}")
            self.dots_frame.config(text="◆ ◆ ◆ ◆ ◆")

            self.splash.update()
            self.splash.after(50, self.animate)
        else:
            self.splash.destroy()


class DataManager:
    """Handles dataset loading and transaction streaming"""

    def __init__(self, csv_file="fraud_demo_dataset_clean.csv"):
        self.csv_file = csv_file
        self.df = None
        self.current_index = 0
        self.load_data()

    def load_data(self):
        """Load CSV if exists"""
        if os.path.exists(self.csv_file) and PANDAS_AVAILABLE:
            try:
                self.df = pd.read_csv(self.csv_file)
                print(f"✓ Loaded dataset: {self.csv_file} ({len(self.df)} rows)")
                return True
            except Exception as e:
                print(f"Error loading CSV: {e}")
        self.df = None
        return False

    def get_next_transaction(self):
        """Get next transaction from CSV or generate"""
        if self.df is not None and len(self.df) > 0:
            if self.current_index >= len(self.df):
                self.current_index = 0

            row = self.df.iloc[self.current_index]
            self.current_index += 1

            return {
                'id': str(row.get('transaction_id', f'TX{random.randint(100000, 999999)}')),
                'timestamp': str(row.get('timestamp', datetime.datetime.now())),
                'amount': float(row.get('amount', random.uniform(10, 10000))),
                'type': str(row.get('transaction_type', 'transfer')),
                'category': str(row.get('merchant_category', 'retail')),
                'location': str(row.get('location', 'Unknown')),
                'device': str(row.get('device_used', 'web')),
                'label': int(row.get('is_fraud', 0))
            }
        else:
            # Improved fraud generation - 15% fraud rate instead of 5%
            is_fraud_flag = random.random() < 0.15
            
            return {
                'id': f'TX{random.randint(100000, 999999)}',
                'timestamp': datetime.datetime.now().isoformat(),
                'amount': random.uniform(5000, 15000) if is_fraud_flag else random.uniform(10, 500),
                'type': random.choice(['withdrawal', 'deposit', 'payment', 'transfer']),
                'category': random.choice(['utilities', 'retail', 'entertainment', 'travel']),
                'location': random.choice(['Dubai', 'Hong Kong', 'Unknown', 'New York', 'London']) if is_fraud_flag else random.choice(['New York', 'London', 'Tokyo', 'Singapore']),
                'device': random.choice(['unknown', 'web', 'atm']) if is_fraud_flag else random.choice(['mobile', 'web']),
                'label': 1 if is_fraud_flag else 0
            }


class MLFraudDetector:
    """Advanced ML-based anomaly detection"""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.confidence_history = []
        self.train_model()

    def train_model(self):
        """Train ML model with better fraud patterns"""
        if SKLEARN_AVAILABLE:
            X_train = []
            y_train = []

            # Normal transactions
            for _ in range(1500):
                amount = random.uniform(10, 500)
                velocity = random.uniform(0.1, 1.5)
                frequency = random.uniform(0, 5)
                X_train.append([amount, velocity, frequency])
                y_train.append(0)

            # Fraudulent transactions
            for _ in range(500):
                amount = random.uniform(5000, 20000)
                velocity = random.uniform(2.0, 4.0)
                frequency = random.uniform(8, 20)
                X_train.append([amount, velocity, frequency])
                y_train.append(1)

            X_train = np.array(X_train)
            y_train = np.array(y_train)

            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)

            self.model = RandomForestClassifier(
                n_estimators=200, max_depth=15, random_state=42, 
                class_weight='balanced', n_jobs=-1
            )
            self.model.fit(X_train_scaled, y_train)
            print("✓ ML model trained with improved fraud patterns")

    def predict_risk(self, transaction):
        """Predict fraud risk with confidence"""
        if self.model and SKLEARN_AVAILABLE:
            amount = transaction['amount']
            velocity = min(amount / 500, 4.0)
            frequency = random.uniform(0, 10)

            X = np.array([[amount, velocity, frequency]])
            X_scaled = self.scaler.transform(X)
            proba = self.model.predict_proba(X_scaled)[0][1]
            confidence = self.model.predict_proba(X_scaled)[0].max()
            
            self.confidence_history.append(confidence)
            if len(self.confidence_history) > 100:
                self.confidence_history.pop(0)
            
            return float(proba), float(confidence)
        else:
            amount_risk = min(transaction['amount'] / 5000, 1.0)
            return amount_risk * 0.8 + random.random() * 0.2, random.random()


class AdvancedFraudEngine:
    """Multi-factor fraud risk assessment - IMPROVED"""

    def __init__(self):
        self.ml_detector = MLFraudDetector()
        self.biometric_profiles = {}
        self.threat_intelligence = self.generate_threats()
        self.fraud_network = defaultdict(list)

    def generate_threats(self):
        """Generate dark web threats"""
        threats = []
        for i in range(15):
            threats.append({
                'id': f'THREAT_{i}',
                'severity': random.choice(['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']),
                'description': f'Suspicious activity pattern detected',
                'timestamp': (datetime.datetime.now() - datetime.timedelta(hours=random.randint(1, 48))).isoformat(),
                'confidence': random.uniform(0.6, 0.99)
            })
        return threats

    def calculate_risk_score(self, transaction):
        """Calculate comprehensive risk score - IMPROVED THRESHOLDS"""
        
        # ML Risk (Higher weight now - 40%)
        ml_risk, ml_confidence = self.ml_detector.predict_risk(transaction)
        
        # Certificate Risk (10%)
        certificate_risk = 0.1 if random.random() > 0.05 else 0.7
        
        # Amount Risk (15%) - More aggressive
        amount_risk = min(transaction['amount'] / 3000, 1.0)
        
        # Velocity Risk (10%)
        velocity_risk = min(transaction['amount'] / 300, 1.0) * 0.5
        
        # Dark Web Risk (15%)
        dark_web_risk = self.calculate_dark_web_risk(transaction)
        
        # Biometric Risk (7%)
        biometric_risk = self.calculate_biometric_risk(transaction)
        
        # Location Risk (3%)
        location_risk = 0.3 if transaction['location'] in ['Dubai', 'Hong Kong', 'Unknown'] else 0.05

        # IMPROVED WEIGHTING
        risk_score = (
            ml_risk * 0.40 +           # Increased from 0.30
            amount_risk * 0.15 +       # Increased from 0.12
            dark_web_risk * 0.15 +
            certificate_risk * 0.10 +  # Decreased from 0.15
            velocity_risk * 0.10 +
            biometric_risk * 0.07 +    # Decreased from 0.12
            location_risk * 0.03
        )

        return min(max(risk_score, 0.0), 1.0), ml_confidence

    def calculate_dark_web_risk(self, transaction):
        """Calculate dark web risk"""
        high_risk_locations = ['Unknown', 'Dubai', 'Hong Kong']
        location_risk = 0.4 if transaction['location'] in high_risk_locations else 0.05
        threat_correlation = 0.3 if random.random() < 0.05 else 0.0
        return min(location_risk + threat_correlation, 1.0)

    def calculate_biometric_risk(self, transaction):
        """Calculate biometric risk"""
        device = transaction['device']
        
        if device not in self.biometric_profiles:
            self.biometric_profiles[device] = {'count': 0, 'patterns': []}

        self.biometric_profiles[device]['count'] += 1
        self.biometric_profiles[device]['patterns'].append(random.random())

        if self.biometric_profiles[device]['count'] < 3:
            return 0.4
        
        if device in ['unknown', 'atm']:
            return 0.5
            
        return 0.1

    def should_block(self, risk_score):
        """LOWER THRESHOLD - was 0.60, now 0.45"""
        return risk_score > 0.45

    def get_risk_level(self, risk_score):
        """Categorize risk level"""
        if risk_score < 0.30:
            return "LOW"
        elif risk_score < 0.45:
            return "MEDIUM"
        else:
            return "HIGH"

    def get_risk_details(self, transaction, risk_score):
        """Get risk breakdown"""
        ml_risk, _ = self.ml_detector.predict_risk(transaction)
        return {
            'ml': ml_risk,
            'amount': min(transaction['amount'] / 3000, 1.0),
            'location': transaction['location'],
            'device': transaction['device'],
            'overall': risk_score
        }


class AdvancedFraudDetectionApp:
    """Main GUI application"""

    def __init__(self, root):
        self.root = root
        self.root.title("🔐 Advanced Certificate Fraud Detection System")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#0f0f1e')

        self.engine = AdvancedFraudEngine()
        self.data_manager = DataManager()

        self.monitoring = False
        self.transactions = []
        self.alerts = []
        self.stats = {
            'total': 0,
            'blocked': 0,
            'approved': 0,
            'average_risk': 0.0,
            'avg_confidence': 0.0,
        }

        self.setup_gui()

    def setup_gui(self):
        """Build GUI"""
        main_frame = tk.Frame(self.root, bg='#0f0f1e')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        control_frame = tk.Frame(main_frame, bg='#1a1a3e', relief=tk.RAISED, bd=2)
        control_frame.pack(fill=tk.X, pady=5)

        title_frame = tk.Frame(control_frame, bg='#1a1a3e')
        title_frame.pack(fill=tk.X, padx=20, pady=15)

        title = tk.Label(title_frame, text="🔐 ADVANCED FRAUD DETECTION DASHBOARD",
                        font=("Arial", 20, "bold"), fg="#00ffff", bg='#1a1a3e')
        title.pack(side=tk.LEFT)

        subtitle = tk.Label(title_frame, text="Real-Time AI-Powered Certificate Analysis",
                           font=("Arial", 11), fg="#00ff88", bg='#1a1a3e')
        subtitle.pack(side=tk.LEFT, padx=50)

        btn_frame = tk.Frame(control_frame, bg='#1a1a3e')
        btn_frame.pack(pady=10, fill=tk.X, padx=20)

        tk.Button(btn_frame, text="▶️  START MONITORING", command=self.start_monitoring,
                 bg='#00ff88', fg='#000', font=("Arial", 11, "bold"), width=18,
                 padx=10, pady=8, relief=tk.RAISED, bd=2).pack(side=tk.LEFT, padx=5)

        tk.Button(btn_frame, text="⏹️  STOP MONITORING", command=self.stop_monitoring,
                 bg='#ff5555', fg='#fff', font=("Arial", 11, "bold"), width=18,
                 padx=10, pady=8, relief=tk.RAISED, bd=2).pack(side=tk.LEFT, padx=5)

        tk.Button(btn_frame, text="📊 EXPORT REPORT", command=self.export_report,
                 bg='#5555ff', fg='#fff', font=("Arial", 11, "bold"), width=18,
                 padx=10, pady=8, relief=tk.RAISED, bd=2).pack(side=tk.LEFT, padx=5)

        tk.Button(btn_frame, text="🔍 ANALYSIS", command=self.show_analysis,
                 bg='#ff9900', fg='#fff', font=("Arial", 11, "bold"), width=18,
                 padx=10, pady=8, relief=tk.RAISED, bd=2).pack(side=tk.LEFT, padx=5)

        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=5)

        trans_frame = tk.Frame(notebook, bg='#0f0f1e')
        notebook.add(trans_frame, text="📊 LIVE TRANSACTIONS")

        self.transaction_text = scrolledtext.ScrolledText(trans_frame, height=15, width=190,
                                                          bg='#0a0a15', fg='#00ff88',
                                                          font=("Courier", 8),
                                                          insertbackground='#00ffff')
        self.transaction_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.transaction_text.tag_config('HIGH', foreground='#ff4444', background='#330000')
        self.transaction_text.tag_config('MEDIUM', foreground='#ffff00', background='#333300')
        self.transaction_text.tag_config('LOW', foreground='#00ff88')

        alert_frame = tk.Frame(notebook, bg='#0f0f1e')
        notebook.add(alert_frame, text="🚨 SECURITY ALERTS")

        self.alert_text = scrolledtext.ScrolledText(alert_frame, height=15, width=190,
                                                    bg='#1a0a0a', fg='#ff6666',
                                                    font=("Courier", 8),
                                                    insertbackground='#ff0000')
        self.alert_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        stats_frame = tk.Frame(notebook, bg='#1a1a2e')
        notebook.add(stats_frame, text="📈 STATISTICS")

        self.stats_label = tk.Label(stats_frame, text="", font=("Courier", 10),
                                   fg='#00ffff', bg='#1a1a2e', justify=tk.LEFT)
        self.stats_label.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

        threat_frame = tk.Frame(notebook, bg='#0f0f1e')
        notebook.add(threat_frame, text="🌐 DARK WEB INTELLIGENCE")

        self.threat_text = scrolledtext.ScrolledText(threat_frame, height=15, width=190,
                                                     bg='#1a0a2e', fg='#ff00ff',
                                                     font=("Courier", 8))
        self.threat_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.display_threats()

        log_frame = tk.Frame(notebook, bg='#0f0f1e')
        notebook.add(log_frame, text="📋 SYSTEM LOGS")

        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, width=190,
                                                  bg='#0a0a15', fg='#888888',
                                                  font=("Courier", 7))
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_system_start()

        status_frame = tk.Frame(main_frame, bg='#000000', relief=tk.SUNKEN, bd=2, height=50)
        status_frame.pack(fill=tk.X, pady=2)

        left_status = tk.Frame(status_frame, bg='#000000')
        left_status.pack(side=tk.LEFT, padx=10, pady=5)

        self.status_label = tk.Label(left_status, text="🟢 Ready",
                                     font=("Arial", 10, "bold"), fg='#00ff88', bg='#000000')
        self.status_label.pack(side=tk.LEFT)

        self.dataset_label = tk.Label(left_status, text="Dataset: Real/Simulation",
                                      font=("Arial", 9), fg='#00ffff', bg='#000000')
        self.dataset_label.pack(side=tk.LEFT, padx=20)

        self.time_label = tk.Label(status_frame, text="", font=("Arial", 9),
                                   fg='#ffff00', bg='#000000')
        self.time_label.pack(side=tk.RIGHT, padx=10, pady=5)

        self.update_time()

    def display_threats(self):
        """Display threats"""
        self.threat_text.delete(1.0, tk.END)
        self.threat_text.insert(tk.END, "🌐 DARK WEB THREAT INTELLIGENCE FEED\n")
        self.threat_text.insert(tk.END, "=" * 180 + "\n\n")
        
        for threat in self.engine.threat_intelligence:
            severity_color = "🔴" if threat['severity'] == 'CRITICAL' else "🟠" if threat['severity'] == 'HIGH' else "🟡"
            self.threat_text.insert(tk.END,
                f"{severity_color} [{threat['severity']}] {threat['id']} | {threat['description']}\n"
                f"   ├─ Confidence: {threat['confidence']:.1%}\n"
                f"   └─ Detected: {threat['timestamp']}\n\n")

    def log_system_start(self):
        """Log startup"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] System initialized\n")
        self.log_text.insert(tk.END, f"[{timestamp}] ML model loaded (200 estimators)\n")
        self.log_text.insert(tk.END, f"[{timestamp}] Detection threshold: 0.45 (IMPROVED)\n")
        self.log_text.insert(tk.END, f"[{timestamp}] Expected fraud rate: ~15%\n")

    def start_monitoring(self):
        """Start monitoring"""
        if not self.monitoring:
            self.monitoring = True
            self.status_label.config(text="🔴 MONITORING ACTIVE")
            self.dataset_label.config(text=f"Dataset: {'Real' if self.data_manager.df is not None else 'Simulation'}")
            threading.Thread(target=self.monitor_loop, daemon=True).start()

    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        self.status_label.config(text="🟢 Ready")

    def monitor_loop(self):
        """Main loop"""
        while self.monitoring:
            tx = self.data_manager.get_next_transaction()
            risk_score, confidence = self.engine.calculate_risk_score(tx)

            self.stats['total'] += 1
            self.stats['average_risk'] = (
                (self.stats['average_risk'] * (self.stats['total'] - 1) + risk_score) /
                self.stats['total']
            )
            self.stats['avg_confidence'] = (
                (self.stats['avg_confidence'] * (self.stats['total'] - 1) + confidence) /
                self.stats['total']
            )

            risk_level = self.engine.get_risk_level(risk_score)

            if self.engine.should_block(risk_score):
                self.stats['blocked'] += 1
                alert_msg = (f"🚨 FRAUD BLOCKED | {tx['id']} | Risk: {risk_score:.1%} | "
                           f"Confidence: {confidence:.1%} | Amount: ${tx['amount']:.2f} | {tx['location']}")
                self.alert_text.insert(tk.END, alert_msg + "\n")
                self.alert_text.see(tk.END)

                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                self.log_text.insert(tk.END, f"[{timestamp}] 🚨 BLOCKED: {tx['id']} (risk: {risk_score:.1%}, amount: ${tx['amount']:.2f})\n")
                self.log_text.see(tk.END)
                
                self.alerts.append({
                    'id': tx['id'],
                    'time': datetime.datetime.now().isoformat(),
                    'risk': risk_score,
                    'amount': tx['amount'],
                    'location': tx['location']
                })
            else:
                self.stats['approved'] += 1

            tx_msg = (f"✓ {tx['id']} | {risk_level:8} | Risk: {risk_score:6.1%} | "
                     f"Conf: {confidence:6.1%} | ${tx['amount']:8.2f} | {tx['location']:12} | {tx['device']:6}")
            
            self.transaction_text.insert(tk.END, tx_msg + "\n", risk_level)
            self.transaction_text.see(tk.END)

            self.update_statistics()
            self.root.update()
            time.sleep(0.8)

    def update_statistics(self):
        """Update stats"""
        block_rate = (self.stats['blocked'] / max(self.stats['total'], 1)) * 100
        
        stats_text = f"""
╔═══════════════════════════════════════════════════════════════════════════════════════╗
║                        📊 REAL-TIME FRAUD DETECTION STATISTICS                       ║
╠═══════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                       ║
║  TRANSACTION METRICS:                                                                ║
║  ├─ Total Processed:        {self.stats['total']:,}                                    ║
║  ├─ Fraud BLOCKED:          {self.stats['blocked']:,} transactions ⚠️                  ║
║  ├─ Approved:               {self.stats['approved']:,} transactions ✅                 ║
║  └─ Block Rate:             {block_rate:.2f}%                                         ║
║                                                                                       ║
║  RISK ANALYSIS:                                                                      ║
║  ├─ Average Risk Score:     {self.stats['average_risk']:.2%}                          ║
║  ├─ ML Confidence:          {self.stats['avg_confidence']:.2%}                        ║
║  ├─ Active Alerts:          {len(self.alerts)}                                        ║
║  └─ Detection Threshold:    45% (IMPROVED)                                           ║
║                                                                                       ║
║  SYSTEM STATUS:                                                                      ║
║  ├─ Monitoring:             {'ACTIVE 🟢' if self.monitoring else 'STANDBY'}          ║
║  ├─ Dataset Mode:           {'Real CSV' if self.data_manager.df is not None else 'Simulation'} ║
║  ├─ ML Model:               200 Estimators (RF Classifier)                           ║
║  └─ Fraud Rate Expected:    ~15% (was 5%)                                            ║
║                                                                                       ║
╚═══════════════════════════════════════════════════════════════════════════════════════╝
        """
        self.stats_label.config(text=stats_text)

    def export_report(self):
        """Export report"""
        filename = f"fraud_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            'timestamp': datetime.datetime.now().isoformat(),
            'summary': self.stats,
            'alerts': self.alerts[:100],
            'transactions_processed': len(self.transactions),
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        messagebox.showinfo("Export Successful", f"Report exported to {filename}")

    def show_analysis(self):
        """Show analysis"""
        analysis = (f"Advanced Fraud Detection Analysis\n"
                   f"================================\n\n"
                   f"Total Transactions: {self.stats['total']}\n"
                   f"Fraud Detection Rate: {(self.stats['blocked'] / max(self.stats['total'], 1)) * 100:.1f}%\n"
                   f"Average Risk Score: {self.stats['average_risk']:.2%}\n\n"
                   f"IMPROVEMENTS MADE:\n"
                   f"• Lowered detection threshold to 45%\n"
                   f"• Increased fraud rate to 15%\n"
                   f"• Better fraud patterns in data\n"
                   f"• Higher ML weighting (40%)\n"
                   f"• Enhanced amount analysis\n"
                   f"• Improved dark web correlation")
        messagebox.showinfo("Advanced Analysis", analysis)

    def update_time(self):
        """Update time"""
        self.time_label.config(text=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.root.after(1000, self.update_time)


if __name__ == "__main__":
    root = tk.Tk()
    
    splash = AdvancedSplashScreen(root)
    root.update()
    time.sleep(0.5)

    app = AdvancedFraudDetectionApp(root)
    root.mainloop()