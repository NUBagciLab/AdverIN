from MedSegDGSSL.utils import Registry, check_availability

EVALUATOR_REGISTRY = Registry('EVALUATOR')


def build_evaluator(cfg, **kwargs):
    avai_evaluators = EVALUATOR_REGISTRY.registered_names()
    check_availability(cfg.TEST.EVALUATOR, avai_evaluators)
    if cfg.VERBOSE:
        print('Loading evaluator: {}'.format(cfg.TEST.EVALUATOR))
    return EVALUATOR_REGISTRY.get(cfg.TEST.EVALUATOR)(cfg, **kwargs)

def build_final_evaluator(cfg, **kwargs):
    avai_evaluators = EVALUATOR_REGISTRY.registered_names()
    check_availability(cfg.TEST.FINAL_EVALUATOR, avai_evaluators)
    if cfg.VERBOSE:
        print('Loading evaluator: {}'.format(cfg.TEST.EVALUATOR))
    return EVALUATOR_REGISTRY.get(cfg.TEST.FINAL_EVALUATOR)(cfg, **kwargs)
