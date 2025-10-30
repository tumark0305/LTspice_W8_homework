
class data_info:
    def __init__(self , _text_input):
        self.text_input = _text_input
        self.line = None
        self.Lx = None
        self.Cx = None
        self.Rx = None
        self.stop_time = None
        self.f0 = None
        self.BW = None
        self.__praser()
        return None
    def __praser(self):
        _variables = self.text_input.split('=')[0]
        _outputs = self.text_input.split('=')[1]
        self.line = int(_variables.split(',')[0])
        self.Lx = float(_variables.split(',')[1])
        self.Cx = float(_variables.split(',')[2])
        self.Rx = float(_variables.split(',')[3])
        self.stop_time = float(_outputs.split(',')[0])
        self.f0 = float(_outputs.split(',')[1])
        self.BW = float(_outputs.split(',')[2])
        return None

class data_coonverter:
    def __init__(self,_file_path:str):
        _f = open(_file_path,'r')
        _data = _f.read()
        _f.close()
        self.data = [data_info(_line) for _line in _data.split('\n')]
        return None
    def csv_by_Cx(self,_file_path:str):
        _output_list = [f"constant:,Lx,{self.data[0].Lx},Rx,{self.data[0].Rx},,,Variable,Cx",f'Cx,stop,f0,BW']
        for _data in self.data:
            _output_list.append(f'{_data.Cx},{_data.stop_time},{_data.f0},{_data.BW}')
        _output_text = '\n'.join(_output_list)
        _f = open(_file_path,'w')
        _f.write(_output_text)
        _f.close()
        return None
    def csv_by_Rx(self,_file_path:str):
        _output_list = [f"constant:,Lx,{self.data[0].Lx},Cx,{self.data[0].Cx},,,Variable,Rx",f'Cx,stop,f0,BW']
        for _data in self.data:
            _output_list.append(f'{_data.Rx},{_data.stop_time},{_data.f0},{_data.BW}')
        _output_text = '\n'.join(_output_list)
        _f = open(_file_path,'w')
        _f.write(_output_text)
        _f.close()
        return None


if __name__ == '__main__':
    _result = data_coonverter('./Circuit/var_C29.txt')
    _result.csv_by_Cx('./Circuit/C29.csv')





