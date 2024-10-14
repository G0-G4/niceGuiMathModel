import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from nicegui import ui


class Model:

    def __init__(self):
        self.T = 30
        self.T1 = 3
        self.a = 3
        self.b = 4
        self.tn = 5
        self.x = np.arange(0, self.T, 0.1)
        self.I = np.vectorize(self._I)
        self.D = np.vectorize(self._D)
        self.q0 = self._q0()
        self.B = self._B()
        self.Q = self._Q()

        # self.fig = go.Figure(go.Scatter(x=self.x, y=self.I(self.x)))
        self.fig = make_subplots(2, 1)
        self.fig.add_trace(go.Scatter(x=self.x, y=self.I(self.x), name='I(t)'), 1, 1)
        self.fig.add_trace(go.Scatter(x=self.x, y=self.D(self.x), name='D(t)'), 2, 1)
        self.fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))

    def _q0(self):
        return self.a * self.T1 + self.b / 2 * self.T1 ** 2

    def _B(self):
        return -self.I(self.T)

    def _Q(self):
        return self.q0 + self.B

    def _I(self, t):
        T = self.T
        T1 = self.T1
        a = self.a
        b = self.b
        tn = self.tn
        if 0 <= t <= tn:
            return a * (T1 - t) + b / 2 * (T1 ** 2 - t ** 2)
        elif tn <= t <= T:
            return self._I(tn) - (a + b * tn) * (t - tn)

    def _D(self, t):
        if t < self.tn:
            return self.a + self.b*t
        return self.a + self.b * self.tn

    def update_plot(self):
        if not self.check_all_set():
            return
        self.q0 = self._q0()
        self.B = self._B()
        self.Q = self._Q()
        self.fig.update_traces(x=self.x, y=self.I(self.x), row=1, col=1)
        self.fig.update_traces(x=self.x, y=self.D(self.x), row=2, col=1)

    def check_all_set(self):
        return self.T and self.T1 and self.tn and self.a and self.b


class App:
    def __init__(self):
        self.tn = None
        self.T = None
        self.T1 = None
        self.model = Model()
        self.plot = None
        self.generate_layout()

    def update_plot(self):
        self.model.update_plot()
        self.plot.update()

    def update_boundaries(self):
        if not (self.T1.value and self.T.value and self.tn.value):
            return
        self.T.min = self.model.T1
        self.tn.min = self.model.T1
        self.tn.max = self.model.T
        self.update_plot()

    def generate_layout(self):
        with (ui.column()):
            with ui.row():
                self.T1 = ui.number(label='T1', min=0).bind_value(self.model, 'T1').on_value_change(self.update_boundaries)
                self.tn = ui.number(label='tn', min=self.model.T1, max=self.model.T).bind_value(self.model, 'tn').on_value_change(self.update_boundaries)
                self.T = ui.number(label='T', min=self.model.T1).bind_value(
                    self.model, 'T').on_value_change(self.update_boundaries)
            with ui.row():
                ui.number(label='a').bind_value(self.model, 'a').on_value_change(self.update_plot)
                ui.number(label='b').bind_value(self.model, 'b').on_value_change(self.update_plot)
        with ui.row():
            self.plot = ui.plotly(self.model.fig)
            ui.number(label='q0').bind_value(self.model, 'q0')
            ui.number(label='B').bind_value(self.model, 'B')
            ui.number(label='Q').bind_value(self.model, 'Q')


app = App()
ui.run()
