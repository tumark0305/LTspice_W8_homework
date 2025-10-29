from LTtoolbox import LTspice
import os,subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

class Auto_spice:
    def __init__(self):
        return None

    def muti_test(self):
        results = [None]*5
        with ThreadPoolExecutor(max_workers=5) as ex:
            for i in range(5):
                fut = ex.submit(_run_one_case, i, CACHE, self.circuit_data)
                futures.append(fut)

            for fut in as_completed(futures):
                idx, stop, f0, BW = fut.result()
                results[idx] = (stop, f0, BW)

        # 彙整輸出（照 idx 排好）
        for i, r in enumerate(results):
            stop, f0, BW = r
            print(f"[summary {i}] stop={stop} , f0={f0} , BW={BW}")
        return None

if __name__ == '__main__':
    _bot = Auto_spice
