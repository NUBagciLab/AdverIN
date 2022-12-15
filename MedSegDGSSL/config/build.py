from MedSegDGSSL.utils import Registry, check_availability

CONFIG_REGISTRY = Registry('CONFIG')

def get_config(trainer_name):
    avai_trainers = CONFIG_REGISTRY.registered_names()
    check_availability(trainer_name, avai_trainers)
    return CONFIG_REGISTRY.get(trainer_name)

