import numpy as np
import plotly.graph_objects as go
from nicegui import ui
from plotly.subplots import make_subplots

WIDTH = "w-20"


def clamp(value, min_value, max_value):
    return max(min(value, max_value), min_value)


class Model:

    def __init__(self):
        self.T = 15
        self.T1 = 3
        self.a = 1
        self.b = 4
        self.p = 15000
        self.c = 10000
        self.s = 0
        self.tn = 6
        self.x = np.arange(0, self.T, 0.1)
        self.I = np.vectorize(self._I)
        self.D = np.vectorize(self._D)
        self.q0 = self._q0()
        self.B = self._B()
        self.Q = self._Q()
        self.V = self._V()
        self.TP = self._TP()
        # self.fig = go.Figure(go.Scatter(x=self.x, y=self.I(self.x)))
        self.fig = make_subplots(2, 1)
        self.fig.add_trace(go.Scatter(x=self.x, y=self.I(self.x), name='I(t)'), 1, 1)
        self.fig.add_trace(go.Scatter(x=[self.tn], y=self.I(self.tn), name='tn'), 1, 1)
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
        elif tn < t <= T:
            return self._I(tn) - (a + b * tn) * (t - tn)

    def _D(self, t):
        if t < self.tn:
            return self.a + self.b * t
        return self.a + self.b * self.tn

    def _V(self):
        It1tn = -(self.T1 - self.tn) ** 2 / 6 * (3 * self.a + self.b * (self.tn + 2 * self.T1))
        ItnT = self._I(self.tn) * (self.T - self.tn) - (self.a + self.b * self.tn) * (self.T - self.tn) ** 2 / 2
        return -self.s * (It1tn + ItnT)

    def _TP(self):
        return (self.p - self.c) * self._Q() - self._V()

    def update(self):
        if not self.check_all_set():
            return
        self.q0 = self._q0()
        self.B = self._B()
        self.Q = self._Q()
        self.V = self._V()
        self.TP = self._TP()
        self.fig.update_traces(x=self.x, y=self.I(self.x), row=1, col=1, selector={'name': 'I(t)'})
        self.fig.update_traces(x=[self.tn], y=self.I(self.tn), row=1, col=1, selector={'name': 'tn'})
        self.fig.update_traces(x=self.x, y=self.D(self.x), row=2, col=1, selector={'name': 'D(t)'})

    def check_all_set(self):
        return self.T is not None and self.T1 is not None and self.tn is not None and self.a is not None and self.b is not None


class App:
    def __init__(self):
        self.tn = None
        self.T = None
        self.T1 = None
        self.model = Model()
        self.plot = None
        self.generate_layout()

    def update_plot(self):
        self.model.update()
        self.plot.update()

    def update_boundaries(self, argument):
        if not (self.T1.value is not None and self.T.value is not None and self.tn.value is not None):
            return
        if argument.sender == self.T1:
            self.T1.value = clamp(argument.value, 0, self.tn.value)
            self.tn.min = self.model.T1
            self.T1.update()
        elif argument.sender == self.tn:
            self.tn.value = clamp(argument.value, self.T1.value, self.T.value)
            self.T.min = self.model.tn
            self.tn.update()
        elif argument.sender == self.T:
            self.T.value = clamp(argument.value, self.tn.value, float('inf'))
            self.tn.max = self.model.T
            self.T.update()
        self.update_plot()

    def generate_layout(self):
        with (ui.column()):
            with ui.card():
                with ui.row():
                    self.T1 = ui.number(label='T1', min=0, max=self.model.tn,
                                        precision=4).classes(WIDTH).tooltip("момент реализации первой партии").bind_value(
                        self.model, 'T1').on_value_change(self.update_boundaries)
                    self.tn = ui.number(label='tn', min=self.model.T1, max=self.model.T,
                                        precision=4).classes(WIDTH).tooltip("момент насыщения").bind_value(self.model,
                                                                                                           'tn').on_value_change(
                        self.update_boundaries)
                    self.T = ui.number(label='T', min=self.model.tn, precision=4).classes(WIDTH).tooltip(
                        "момент реализации всего товарa").bind_value(
                        self.model, 'T').on_value_change(self.update_boundaries)
                with ui.row():
                    ui.number(label='a', precision=4).classes(WIDTH).bind_value(self.model, 'a').on_value_change(
                        self.update_plot)
                    ui.number(label='b', precision=4).classes(WIDTH).bind_value(self.model, 'b').on_value_change(
                        self.update_plot)
                    ui.number(label='p', precision=4, step=100).classes(WIDTH).tooltip('цена покупателя').bind_value(self.model,
                                                                                                           'p').on_value_change(
                        self.update_plot)
                    ui.number(label='c', precision=4, step=100).classes(WIDTH).tooltip('цена закупки').bind_value(self.model,
                                                                                                        'c').on_value_change(
                        self.update_plot)
                    ui.number(label='s', precision=4, step=100).classes(WIDTH).tooltip(
                        'издержка на единицу товара при дефиците').bind_value(self.model,
                                                                              's').on_value_change(
                        self.update_plot)
        with ui.row():
            self.plot = ui.plotly(self.model.fig)
            with ui.card():
                with ui.grid():
                    ui.label('q0 объем первой партии')
                    ui.label().bind_text(self.model, 'q0', backward=lambda x: round(x, 4))

                    ui.label('B максимальный уровень дефицита')
                    ui.label().bind_text(self.model, 'B',
                                                        backward=lambda x: round(x, 4))

                    ui.label('Q общий объем заказа')
                    ui.label().bind_text(self.model, 'Q',
                                                        backward=lambda x: round(x, 4))

                    ui.label('V потери связанные с дефицитом')
                    ui.label().bind_text(self.model, 'V',
                                                        backward=lambda
                                                            x: round(x, 4))

                    ui.label('TP общая прибыль')
                    ui.label().bind_text(self.model, 'TP',
                                                        backward=lambda x: round(x,
                                                                                 4))


app = App()
ui.run()
