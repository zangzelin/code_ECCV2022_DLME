import numpy as np

class Nushadular():
    def __init__(
        self,
        nu_start=0.001,
        nu_end=100,
        epoch_start=300,
        epoch_end=750,
        ) -> None:
        super().__init__()
        
        if nu_end < 0:
            nu_end = nu_start

        s = np.log10(nu_start)
        e = np.log10(nu_end)
        self.vListForEpoch = np.concatenate([
            np.zeros((epoch_start, )) + 10**s,
            np.logspace(s, e, epoch_end-epoch_start),
            np.zeros((17001, )) + 10**e,
        ])
    def Getnu(self, epoch):
        return self.vListForEpoch[epoch]


# class Nushadular():
#     def __init__(
#         self,
#         nu_start=0.001,
#         nu_end=100,
#         epoch_start=300,
#         epoch_end=750,
#         ) -> None:
#         super().__init__()

#         s = nu_start
#         e = nu_end
#         self.vListForEpoch = np.concatenate([
#             np.zeros((epoch_start, )) + s,
#             np.linspace(s, e, epoch_end-epoch_start),
#             np.zeros((17001, )) + e,
#         ])
#         # import matplotlib.pyplot as plt

#         # plt.plot(self.vListForEpoch)
#         # plt.savefig('dfsdf.png')
#         # print(self.vListForEpoch)
#         # input()
#     def Getnu(self, epoch):
#         return self.vListForEpoch[epoch]