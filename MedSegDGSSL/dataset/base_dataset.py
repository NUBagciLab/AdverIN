'''
Basic Dataloader which not sp
Data should be organized using Processes format, which likes
```
Data
----Domain1
--------Case01.pnz
--------Case02.pnz
----Domain2
--------Case01.pnz
--------Case02.pnz
```

'''

import os


class Datum(object):
    """
    Data domain basic informations
    Args:
        data_dir: dir to the data
    """
    def __init__(self, data_dir:str):
        super().__init__()
        self.data_dir = data_dir
        self.domains = sorted([domain for domain in os.listdir(self.data_dir) \
                                if not os.path.isfile(os.path.join(self.data_dir, domain))])
        self.domains_dict = {domain:idx for idx, domain in enumerate(self.domains)}
        self.domain_num = len(self.domains)
    
    @property
    def get_datapath(self):
        return self.data_dir
    
    @property
    def get_domains(self):
        return self.domains
    
    @property
    def get_domain_num(self):
        return self.domain_num

    def check_available(self, domain):
        return domain in self.domains
    
    def get_domain_id(self, domain):
        return self.domains_dict[domain]


class DatasetBase(object):
    """
    To define the basic Database, split the data according to the domain
    Args:
        data_dir: the dir to the dat
        num_classes: the output classes
        train_domains: domain for training
        unlabel_domains: domain for unlabeling data
        test_domains: domain for testing
    
    Output:
        train_files: list[{"images":img, "labels":seg, "domain":domain}..]
        unlabel_files: list[{"images":img, "domain":domain}..]
        test_files: list[{"images":img, "labels":seg, "domain":domain}..]
    """
    def __init__(self, data_dir:str, num_classes:int=2,
                       train_domains:list=None,
                       unlabel_domains:list=None,
                       val_domains:list=None,
                       test_domains:list=None) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.datum = Datum(data_dir=self.data_dir)
        self.train_domains = train_domains
        self.unlabel_domains = unlabel_domains
        self.val_domains = val_domains
        self.test_domains = test_domains
        self.num_classes = num_classes
        self.train_x, self.train_u, self.val, self.test = None, None, None, None
        self.check_domain()
        self.get_files()

    def check_domain(self):
        for domain in self.train_domains:
            print(domain)
            print(self.datum.get_domains)
            if not self.datum.check_available(domain):
                raise ValueError(f"Train domain {domain} not in the domain list {self.datum.get_domains}")

        if self.unlabel_domains is not None:
            for domain in self.unlabel_domains:
                if not self.datum.check_available(domain):
                    raise ValueError(f"Unlabel domain {domain} not in the domain list {self.datum.get_domains}")

        if self.val_domains is not None:
            for domain in self.val_domains:
                if not self.datum.check_available(domain):
                    raise ValueError(f"Validation domain {domain} not in the domain list {self.datum.get_domains}")

        for domain in self.test_domains:
            if not self.datum.check_available(domain):
                raise ValueError(f"Test domain {domain} not in the domain list {self.datum.get_domains}")

    def generate_domain_data_list(self, domain_list):
        data_list= []
        for domain in domain_list:
            path_to_file = os.path.join(self.data_dir, domain)
            domain_idx = self.datum.get_domain_id(domain)

            temp_file = [name for name in os.listdir(path_to_file) if name.endswith('.npz')]
            temp_list = [{"data": os.path.abspath(os.path.join(self.data_dir, domain, item)), "domain": domain_idx} for item in temp_file]
            data_list.extend(temp_list)
        return data_list

    def get_files(self):
        self.train_x = self.generate_domain_data_list(self.train_domains)
        self.test = self.generate_domain_data_list(self.test_domains)
        print(self.train_x)

        if self.val_domains is not None:
            self.val = self.generate_domain_data_list(self.val_domains)

        if self.unlabel_domains is not None:
            self.train_u = self.generate_domain_data_list(self.unlabel_domains)
