import itertools, subprocess, pathlib, re, csv, time

# === 路徑設定 ===
EXE = r"C:\Program Files\LTC\LTspiceXVII\XVIIx64.exe"   # LTspice 可執行檔
ASC = pathlib.Path(r"D:\proj\rlc.asc")                  # 你的 schematic
CTRL = ASC.with_name("control.inc")                     # 會被 .include 的參數檔
LOG  = ASC.with_suffix(".log")                          # LTspice 輸出的 .log

# === 你要掃的參數 ===
Cx_list = ["200p", "300p", "470p"]
Lx_list = ["50u", "70u", "100u"]
Rx_list = ["200", "500", "1000"]

def write_control_inc(Cx, Lx, Rx):
    CTRL.write_text(f".param Cx={Cx}\n.param Lx={Lx}\n.param Rx={Rx}\n", encoding="utf-8")

def run_ltspice():
    # -b: batch, -Run: 執行，-ascii: 讓 .raw/.log 易讀
    subprocess.run([EXE, "-b", "-Run", "-ascii", str(ASC)], check=True)

def parse_meas_log(log_path):
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    def grab(name):
        m = re.search(r"\b{name}\s*=\s*([Ee0-9\.\+\-]+)", text)
        return float(m.group(1)) if m else None
    return {
        "peak":  grab("peak"),
        "f0":    grab("f0"),
        "f_low": grab("f_low"),
        "f_high":grab("f_high"),
        "BW":    grab("BW"),
        "Q":     grab("Q"),
    }

rows = []
for Cx, Lx, Rx in itertools.product(Cx_list, Lx_list, Rx_list):
    print(f"Run: Cx={Cx}, Lx={Lx}, Rx={Rx}")
    write_control_inc(Cx, Lx, Rx)
    run_ltspice()
    for _ in range(20):
        if LOG.exists():
            break
        time.sleep(0.1)

    meas = parse_meas_log(LOG)
    meas.update({"Cx": Cx, "Lx": Lx, "Rx": Rx})
    rows.append(meas)

# 寫 CSV
out_csv = ASC.with_suffix(".csv")
with out_csv.open("w", newline="", encoding="utf-8") as f:
    fieldnames = ["Cx","Lx","Rx","f0","BW","Q","f_low","f_high","peak"]
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader(); w.writerows(rows)

print("res", out_csv)

