from keras.optimizers import SGD, RMSprop, Adam, Adadelta, Adagrad, Adamax, Nadam
from architectures.weightnorm import SGDWithWeightnorm, AdamWithWeightnorm


def select(optimizer, initial_lr, clipvalue=-1, clipnorm=-1):

    if optimizer == 'SGD':
        if clipnorm >= 0:
            return SGD(lr=initial_lr, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=clipnorm)
        elif clipvalue >= 0:
            return SGD(lr=initial_lr, decay=1e-6, momentum=0.9, nesterov=True, clipvalue=clipvalue)
        else:
            return SGD(lr=initial_lr, decay=1e-6, momentum=0.9, nesterov=True)
    elif optimizer == 'SGDWithWeightnorm':
        if clipnorm >= 0:
            return SGDWithWeightnorm(lr=initial_lr, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=clipnorm)
        elif clipvalue >= 0:
            return SGDWithWeightnorm(lr=initial_lr, decay=1e-6, momentum=0.9, nesterov=True, clipvalue=clipvalue)
        else:
            return SGDWithWeightnorm(lr=initial_lr, decay=1e-6, momentum=0.9, nesterov=True)
    elif optimizer == 'RMSprop':
            return RMSprop(lr=initial_lr, rho=0.9, epsilon=None, decay=0.0)
    elif optimizer == 'Adagrad':
        if clipnorm >= 0:
            return Adagrad(lr=initial_lr, epsilon=None, decay=0.0, clipnorm=clipnorm)
        elif clipvalue >= 0:
            return Adagrad(lr=initial_lr, epsilon=None, decay=0.0, clipvalue=clipvalue)
        else:
            return Adagrad(lr=initial_lr, epsilon=None, decay=0.0)
    elif optimizer == 'Adadelta':
        if clipnorm >= 0:
            return Adadelta(lr=initial_lr, rho=0.95, epsilon=None, decay=0.0, clipnorm=clipnorm)
        elif clipvalue >= 0:
            return Adadelta(lr=initial_lr, rho=0.95, epsilon=None, decay=0.0, clipvalue=clipvalue)
        else:
            return Adadelta(lr=initial_lr, rho=0.95, epsilon=None, decay=0.0)
    elif optimizer == 'Adam':
        if clipnorm >= 0:
            return Adam(lr=initial_lr, beta_1=0.5, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False,
                        clipnorm=clipnorm)
        elif clipvalue >= 0:
        # clipvalue: Gradients will be clipped when their absolute value exceeds this value
            return Adam(lr=initial_lr, beta_1=0.5, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False, clipvalue=clipvalue)
        else:
            return Adam(lr=initial_lr, beta_1=0.5, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    elif optimizer == 'AdamWithWeightnorm':
        if clipnorm >= 0:
            return AdamWithWeightnorm(lr=initial_lr, beta_1=0.5, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False,
                        clipnorm=clipnorm)
        elif clipvalue >= 0:
        # clipvalue: Gradients will be clipped when their absolute value exceeds this value
            return AdamWithWeightnorm(lr=initial_lr, beta_1=0.5, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False, clipvalue=clipvalue)
        else:
            return AdamWithWeightnorm(lr=initial_lr, beta_1=0.5, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    elif optimizer == 'Adamax':
        if clipnorm >= 0:
            return Adamax(lr=initial_lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, clipnorm=clipnorm)
        elif clipvalue >= 0:
            return Adamax(lr=initial_lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, clipvalue=clipvalue)
        else:
            return Adamax(lr=initial_lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    elif optimizer == 'Nadam':
        if clipnorm >= 0:
            return Nadam(lr=initial_lr, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004,
                         clipnorm=clipnorm)
        elif clipvalue >= 0:
            return Nadam(lr=initial_lr, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004, clipvalue=clipvalue)
        else:
            return Nadam(lr=initial_lr, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)