

class DataParams(object):
    def __init__(self, batch=4, augmentations=[], sequenceLength=[1,1], randSeqOffset=False,
                dataSize=[128,64], dimension=2, simFields=[], simParams=[], normalizeMode=""):
        self.batch          = batch             # batch size
        self.augmentations  = augmentations     # used data augmentations
        self.sequenceLength = sequenceLength    # number of simulation frames in one sequence
        self.randSeqOffset  = randSeqOffset     # randomize sequence starting frame
        self.dataSize       = dataSize          # target data size for scale/crop/cropRandom transformation
        self.dimension      = dimension         # number of data dimension
        self.simFields      = simFields         # which simulation fields are added (vel is always used) from ["dens", "pres"]
        self.simParams      = simParams         # which simulation parameters are added from ["rey", "mach"]
        self.normalizeMode  = normalizeMode     # which mean and std values from different data sets are used in normalization transformation

    @classmethod
    def fromDict(cls, d:dict):
        p = cls()
        p.batch          = d.get("batch",            -1)
        p.augmentations  = d.get("augmentations",    [])
        p.sequenceLength = d.get("sequenceLength",   [])
        p.randSeqOffset  = d.get("randSeqOffset",    False)
        p.dataSize       = d.get("dataSize",         -1)
        p.dimension      = d.get("dimension",        -1)
        p.simFields      = d.get("simFields",        [])
        p.simParams      = d.get("simParams",        [])
        p.normalizeMode  = d.get("normalizeMode",    "")
        return p

    def asDict(self) -> dict:
        return {
            "batch"          : self.batch,
            "augmentations"  : self.augmentations,
            "sequenceLength" : self.sequenceLength,
            "randSeqOffset"  : self.randSeqOffset,
            "dataSize"       : self.dataSize,
            "dimension"      : self.dimension,
            "simFields"      : self.simFields,
            "simParams"      : self.simParams,
            "normalizeMode"  : self.normalizeMode,
        }
