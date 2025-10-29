import os
class subLTspice:
    def __init__(self,_location:str,_file_name:str,_data_pack:list[float],_circuit_data):
        self.location = _location
        self.file_name = _file_name
        if len(_data_pack) != 3:
            raise ValueError(f'{len(_data_pack) = }')
        self.Lx = _data_pack[0]
        self.Cx = _data_pack[1]
        self.Rx = _data_pack[2]
        self.circuit_data = self.__generate(_circuit_data)
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
        _f = open(f'{self.location}/{self.file_name}',"w" ,encoding="utf-8")
        _f.write(self.circuit_data)
        _f.close()
        return None
    def run(self):
        return None


class LTspice:
    process_area = "./Circuit"
    location = f"{process_area}/main.asc"
    cache = f"{process_area}/Cache"
    def __init__(self):
        self.__read_file()
        return None
    def __read_file(self)->str:
        _f = open(self.location,"r" ,encoding="utf-8")
        self.circuit_data = _f.read()
        _f.close()
        return None
    def single_test(self):
        os.makedirs(self.cache,exist_ok=True)
        _circuit = subLTspice(self.cache , "test1.asc",[1e-6,1e-6,1e-6],self.circuit_data)
        _circuit.new_file()
        return None

if __name__ == '__main__':
    circuit = LTspice()
    circuit.single_test()
    pass



