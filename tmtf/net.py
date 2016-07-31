import theano as th
import theano.tensor as T

import Antipasti.netkit as nk
import Antipasti.netarchs as na
import Antipasti.archkit as ak
import Antipasti.netools as ntl
import Antipasti.netrain as nt
import Antipasti.backend as A

__doc__ = """Model Zoo"""

# Define shortcuts
# Convlayer with ELU
cl = lambda fmapsin, fmapsout, kersize: nk.convlayer(fmapsin=fmapsin, fmapsout=fmapsout, kersize=kersize,
                                                     activation=ntl.elu())

clv = lambda fmapsin, fmapsout, kersize: nk.convlayer(fmapsin=fmapsin, fmapsout=fmapsout, kersize=kersize,
                                                      activation=ntl.elu(), convmode='valid')

# Convlayer without activation
cll = lambda fmapsin, fmapsout, kersize: nk.convlayer(fmapsin=fmapsin, fmapsout=fmapsout, kersize=kersize)

# Strided convlayer with ELU (with autopad)
scl = lambda fmapsin, fmapsout, kersize, padding=None: nk.convlayer(fmapsin=fmapsin, fmapsout=fmapsout,
                                                                    kersize=kersize,
                                                                    stride=[2, 2], activation=ntl.elu(),
                                                                    padding=padding)

# Strided 3x3 pool layerlayertrain or Antipasti.netarchs.layertrainyard
spl = lambda: nk.poollayer(ds=[3, 3], stride=[2, 2], padding=[1, 1])

# Batch-norm layer
bn = lambda: nk.batchnormlayer(2, 0.9)

# Softmax
sml = lambda: nk.softmax(dim=2)

# Dropout layer
drl = lambda p=0.5: nk.noiselayer(noisetype='binomial', p=p)


def _build_simple(modelconfig=None):
    """Make network."""
    if modelconfig is None:
        numout = 5
    else:
        numout = modelconfig['numout'] if 'numout' in modelconfig.keys() else 5

    # Build
    network = spl() + scl(5, 32, [9, 9]) + drl() + scl(32, 64, [9, 9]) + spl() + cl(64, 128, [5, 5]) + \
              drl() + scl(128, 256, [5, 5]) + \
              cl(256, 512, [5, 5]) + spl() + cl(512, 512, [3, 3]) + drl() + scl(512, 512, [3, 3]) + \
              clv(512, 1024, [8, 8]) + drl() + cl(1024, 512, [1, 1]) + drl() + cl(512, 256, [1, 1]) + \
              drl() + cl(256, 128, [1, 1]) + \
              cl(128, 64, [1, 1]) + drl() + cl(64, 16, [1, 1]) + drl() + cll(16, numout, [1, 1])

    # Build graph
    network.feedforward()
    # Return
    return network


def simple(modelconfig=None):
    """
    Build a simple model.

    :type modelconfig: dict
    :param modelconfig: Model configuration.
    """

    # Build network
    network = _build_simple(modelconfig)
    # Build target network
    targetnetwork = _build_simple(modelconfig)

    # Compile inference function for network
    network.classifier = A.function(inputs=[network.x], outputs=network.y, allow_input_downcast=True)
    # Compile inference function for target network
    targetnetwork.classifier = A.function(inputs=[targetnetwork.x], outputs=targetnetwork.y, allow_input_downcast=True)

    # Compile trainer for network
    # Redefine target to a scalar
    network.yt = T.vector('model-yt:{}'.format(id(network)))
    # Compute loss and cost
    # network.y.shape = (bs, numout, 1, 1). Compute mean along all axes.
    network.L = ((T.max(T.flatten(network.y, outdim=2), axis=1) - network.yt)**2).mean()
    network.baggage["l2"] = nt.lp(network.params, [(2, 0.0)])
    network.C = network.L + network.baggage["l2"]
    # Compute gradients
    network.dC = T.grad(network.C, wrt=network.params, disconnected_inputs='warn')
    # Get updates
    network.getupdates(method='rmsprop', learningrate=0.0005, rho=0.9)
    # Compile trainer
    network.classifiertrainer = A.function(inputs=[network.x, network.yt], outputs={'C': network.C, 'L': network.L},
                                           updates=network.updates, allow_input_downcast=True)

    # Make update function for targetnetwork and add to its baggage
    def updatetargetparams(params, decay=0.9):
        curparams, newparams = targetnetwork.params, params
        for curparam, newparam in zip(curparams, newparams):
            paramupdate = decay * curparam.get_value() + (1 - decay) * newparam.get_value()
            curparam.set_value(paramupdate)

    targetnetwork.baggage["updatetargetparams"] = updatetargetparams
    # Set target networks param
    targetnetwork.params = network.params

    return network, targetnetwork
