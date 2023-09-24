from easydict import EasyDict
import json


class JsonConfigFileManager:
    """
    json 설정 파일 관리
    """
    def __init__(self, file_path):
        self.values = EasyDict()
        if file_path:
            self.file_path = file_path
            self.reload()
    
    def reload(self):
        """
        설정 리셋, 설정파일 재로드
        """
        self.clear()
        if self.file_path:
            with open(self.file_path, 'r') as f:
                self.values.update(json.load(f))
        
    def clear(self):
        """
        설정 리셋
        """
        self.values.clear()
    
    def update(self, in_dict):
        """
        기존 설정에 새로운 설정 업데이트
        """
        for (k1, v1) in in_dict.items():
            self.values[k1] = v1
            if isinstance(v1, dict):
                for (k2, v2) in v1.items():
                    self.values[k1][k2] = v2

    def export(self, save_file_name):
        """
        설정값을 json 파일로 저장
        """
        if save_file_name:
            with open(save_file_name, 'w') as f:
                json.dump(dict(self.values), f, indent=4)




if __name__ == '__main__':
    conf = JsonConfigFileManager('config/config.json')
    print(conf.values)
    updates = {"model": { 
                    "n_layers": 6, 
                    "n_position": 200, 
                    "d_model": 256, 
                    "d_ff": 2048, 
                    "n_head": 8,  
                    "dropout_p": 0.1}, 
                "train": {
                    "batch_size": 30, 
                    "max_epoch": 18,
                    "step_batch": 26, 
                    "beta1": 0.9, 
                    "beta2": 0.98, 
                    "warmup": 4000, 
                    "eval_interval": 10000,  
                    "smoothing": 0.1}}
    conf.update(updates)
    print(conf.values)
    print(conf.values.model.d_model)
    conf.export('config/config.json')

    conf = JsonConfigFileManager('config/config_test.json')
    print(conf.values)
    updates = {"model": { \
                    "n_layers": 2, 
                    "n_position": 200, 
                    "d_model": 4, 
                    "d_ff": 7, 
                    "n_head": 2,  
                    "dropout_p": 0.1}, 
                "train": {
                    "batch_size": 3, 
                    "step_batch": 5,
                    "max_epoch": 2, 
                    "beta1": 0.9, 
                    "beta2": 0.98, 
                    "warmup": 5, 
                    "eval_interval": 10,  
                    "smoothing": 0.1}}
    conf.update(updates)
    print(conf.values)
    print(conf.values.model.d_model)
    conf.export('config/config_test.json')
