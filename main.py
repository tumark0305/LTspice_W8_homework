from LTtoolbox import LTspice

task_list = [{'file_name': f"case{i}", 'Lx': 50e-6, 'Cx': 300e-12, 'Rx': 1e2 + i*1e2}    for i in range(10)]

class Auto_spice:
    def __init__(self):
        self.circuit = LTspice()
        self.result = None
        self.text = ""
        return None
    def run(self):
        self.result = self.circuit.run_multithread(task_list)
        self.result.sort(key=lambda x: x[0])
        return None
    def save(self):
        _list_text = []
        for _idx in range(len(task_list)):
            _list_text.append(f"{_idx},{task_list[_idx]['Lx']},{task_list[_idx]['Cx']},{task_list[_idx]['Rx']}={ self.result[_idx][1][0]},{ self.result[_idx][1][1]},{ self.result[_idx][1][2]}")
        self.text = "\n".join(_list_text)
        _f = open(f"./Circuit/result.txt","w")
        _f.write(self.text)
        _f.close()
        return None



if __name__ == '__main__':
    _bot = Auto_spice()
    _bot.run()
    _bot.save()

    
    
