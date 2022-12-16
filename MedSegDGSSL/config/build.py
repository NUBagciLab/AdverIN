from MedSegDGSSL.utils import Registry, check_availability

CONFIG_REGISTRY = Registry('CONFIG')

def get_config(trainer_name):
    avai_trainers = CONFIG_REGISTRY.registered_names()
    # If not use specify training, just use the default setting
    if trainer_name not in avai_trainers:
        trainer_name = 'default'
    return CONFIG_REGISTRY.get(trainer_name)()
