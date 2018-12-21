import numpy as np
# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        self.fit = None
        self.best_fit = None
        ## calculate the best_fit according to the history fit
        self.fit_list = []
        self.first_time = True
        self.cache_size = 10

    def add_fit(self, fit):
        self.fit_list.append(fit)
        if len(self.fit_list) > self.cache_size:
            self.fit_list.pop(0)

    def get_best_fit(self):
        a = []
        b = []
        c = []

        res = np.copy(self.fit_list[0])
        for fit in self.fit_list:
            a.append(fit[0])
            b.append(fit[1])
            c.append(fit[2])

        res[0] = np.mean(a)
        res[1] = np.mean(b)
        res[2] = np.mean(c)

        return res

