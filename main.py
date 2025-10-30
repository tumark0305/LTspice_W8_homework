from LTtoolbox import LTspice
import time

TASK = [{'file_name': f"case{i}", 'Lx': 70e-6, 'Cx': 10e-12+i*1e-12, 'Rx': 1000}    for i in range(290)]
TASK1 = [{'file_name': f"case{i}", 'Lx': 70e-6, 'Cx': 300e-12, 'Rx': 1+i}    for i in range(1000)]

class Auto_spice:
    def __init__(self,task_list):
        self.task_list = task_list
        self.circuit = LTspice()
        self.result = None
        self.text = ""
        return None
    def run(self):
        self.result = self.circuit.run_multithread(self.task_list)
        self.result.sort(key=lambda x: x[0])
        return None
    def save(self,file_name:str):
        _list_text = []
        for _idx in range(len(self.task_list)):
            _list_text.append(f"{_idx},{self.task_list[_idx]['Lx']},{self.task_list[_idx]['Cx']},{self.task_list[_idx]['Rx']}={ self.result[_idx][1][0]},{ self.result[_idx][1][1]},{ self.result[_idx][1][2]}")
        self.text = "\n".join(_list_text)
        _f = open(f"./Circuit/{file_name}.txt","w")
        _f.write(self.text)
        _f.close()
        return None



if __name__ == '__main__':
    _bot = Auto_spice(TASK)
    _bot.run()
    _bot.save('var_C290')

    del _bot
    time.sleep(20)
    _bot = Auto_spice(TASK1)
    _bot.run()
    _bot.save('var_R1000')

    
    
