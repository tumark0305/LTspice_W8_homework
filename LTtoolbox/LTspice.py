import os,subprocess
from ltspice import Ltspice
from LTtoolbox import calculator
from concurrent.futures import ThreadPoolExecutor, as_completed

DIR = "./Circuit"
location = f"{DIR}/main.asc"
CACHE = f"{DIR}/Cache"
PROGRAM = f"C:/My Programs/XVIIx64/XVIIx64.exe"

class subLTspice:
    def __init__(self,_file_name:str,_data_pack:list[float],_circuit_data):
        self.file_name = _file_name
        if len(_data_pack) != 3:
            raise ValueError(f'{len(_data_pack) = }')
        self.Lx = _data_pack[0]
        self.Cx = _data_pack[1]
        self.Rx = _data_pack[2]
        self.circuit_data = self.__generate(_circuit_data)
        self.result = []
        return None
    def __generate(self,_circuit_data)->str:
        _parameter_list = [_parameter for _parameter in _circuit_data.split('\n') if ".param" in _parameter]
        if len(_parameter_list) != 1:
            raise ValueError(f'_parameter_list contain more than 1 command:\n{_parameter_list}')
        _output_list = _circuit_data.split('\n')
        for _idx in range(len(_output_list)):
            if ".param" in _output_list[_idx]:
                _parameter__command = _output_list[_idx].split(".param ")
                _new_parameter__command = _parameter__command.copy()
                _new_parameter__command[1] = f"Lx={self.Lx} Cx={self.Cx} Rx={self.Rx}"
                _new_command = ".param ".join(_new_parameter__command)
                _output_list[_idx] = _new_command
        _output = "\n".join(_output_list)
        return _output
    def new_file(self):
        _f = open(f'{CACHE}/{self.file_name}.asc',"w" ,encoding="utf-8")
        _f.write(self.circuit_data)
        _f.close()
        return None
    def run(self):
        #subprocess.run([PROGRAM, "-b", "-Run", "-ascii", f'{CACHE}/{self.file_name}.asc'], check=True)
        subprocess.run([PROGRAM, "-b", "-Run", f'{CACHE}/{self.file_name}.asc'], check=True)
        return None
    def get_result(self):
        lt = Ltspice(f'{CACHE}/{self.file_name}.raw')
        lt.parse()
        x = lt.get_time()
        Vout = lt.get_data('V(out)')
        _cal = calculator(x,Vout)
        if _cal.stop_resonate_time is not None:
            _cal.FFT()
            self.result = [_cal.stop_resonate_time , _cal.F0 , _cal.BW]
        else:
            self.result = [0.0 , 0.0 , 0.0]
        return None


class LTspice:
    def __init__(self):
        os.makedirs(CACHE,exist_ok=True)
        self.__read_file()
        return None
    def __read_file(self)->str:
        _f = open(location,"r" ,encoding="utf-8")
        self.circuit_data = _f.read()
        _f.close()
        return None
    def thread(self,_file_name:str,_Lx=70e-6,_Cx=300e-12,_Rx=1e3):
        _circuit = subLTspice(_file_name,[_Lx,_Cx,_Rx],self.circuit_data)#LCR
        _circuit.new_file()
        _circuit.run()
        _circuit.get_result()
        return _circuit.result
    def run_multithread(self, tasks: list, max_workers: int = 5):
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(
                    self.thread,
                    task['_file_name'],
                    task.get('_Lx', 70e-6),
                    task.get('_Cx', 300e-12),
                    task.get('_Rx', 1e3)
                ): idx
                for idx, task in enumerate(tasks)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    res = future.result()
                except Exception as e:
                    print(f"[Thread {idx}] Error:", e)
                    res = None
                results.append((idx, res))

        return results

if __name__ == '__main__':
    circuit = LTspice()
    circuit.single_test()
    circuit.run_multithread()
    pass



