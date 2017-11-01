try:
    from PyQt4 import QtGui, QtCore
    from PyQt2.QtGui import QToolBar, QSlider
except ImportError:
    from PyQt5 import QtGui, QtCore
    from PyQt5.QtWidgets import QToolBar, QSlider
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
import seaborn as sns
sns.set_style('whitegrid')


class ScrollPlotter:
    """Base class for plots scrolling with a QT scrollbar."""
    scale = 1
    max_page_steps = 40

    def __init__(self, time_size, duration, t_step, decimate):
        self.time_size = time_size
        self.t_step = t_step
        self.duration = duration
        self.duration_steps = duration // t_step
        self.decimate = decimate
        self.num_points = int(self.duration_steps // decimate)
        self.page_step = 1
        if self.time_size / self.duration_steps > self.max_page_steps:
            self.page_step = int(np.ceil(self.time_size / self.duration_steps /
                                 self.max_page_steps))
        self.scroll_step = 1 / 4
        self.smin = 0
        self.smax = self.time_size

        self.create_figure()
        self.fig.patch.set_facecolor("white")
        #self.canvas.setStyleSheet("background-color:transparent;")
        # Retrive the QMainWindow used by current figure and add a toolbar
        # to host the new widgets
        QMainWin = self.fig.canvas.parent()
        toolbar = QToolBar(QMainWin)
        #QMainWin.addToolBar(QtCore.Qt.BottomToolBarArea, toolbar)
        QMainWin.addToolBar(QtCore.Qt.TopToolBarArea, toolbar)
        # Create the slider and spinbox for x-axis scrolling in toolbar
        self.set_slider(toolbar)
        self.init_plot()
        self.update()

    def set_slider(self, parent):
        self.slider = QSlider(QtCore.Qt.Horizontal, parent=parent)
        self.slider.setTickPosition(QSlider.TicksAbove)
        self.slider.setTickInterval((self.smax - self.smin) / 10. * self.scale)

        self.slider.setMinimum(self.smin * self.scale)
        if (self.smax - self.duration_steps) > 0:
            self.slider.setMaximum((self.smax - self.duration_steps) * self.scale)
        else:
            self.slider.setMaximum(self.smax * self.scale)
        self.slider.setSingleStep(self.duration_steps * self.scale * self.scroll_step)
        self.slider.setPageStep(self.duration_steps * self.scale * self.page_step)
        self.slider.setValue(self.smin * self.scale)  # set the initial position
        self.slider.valueChanged.connect(self.slider_changed)
        parent.addWidget(self.slider)

    def slider_changed(self, pos):
        pos /= self.scale
        slice_ = (pos, pos + self.duration_steps, self.decimate)
        self.update(slice_)

    def create_figure(self):
        pass

    def init_plot(self):
        pass


    def update(self, slice_=None):
        pass


class EmissionPlotter(ScrollPlotter):
    def __init__(self, S, particles=None, color_pop=True,
                 duration=0.01, decimate=100):
        if particles is None:
            particles = range(S.num_particles)
        self.particles = particles
        self.color_pop = color_pop
        self.S = S
        super().__init__(S.n_samples, duration, S.t_step, decimate)

    def create_figure(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 6))

    def init_plot(self):
        fig, ax = self.fig, self.ax
        self.title_suffix = "Total: %.1f s, Visualized: %.2f ms" % (
            self.S.t_step * self.S.n_samples, self.duration * 1e3)

        ax.set_xlabel("Time (ms)")

        pal = sns.color_palette()
        colors = pal[1:]
        par_counts = [c[1] for c in self.S.particles.diffusion_coeff_counts]

        em_y = np.zeros(self.num_points)
        em_x = np.arange(self.num_points) * self.S.t_step * self.decimate * 1e3
        lines_em = []
        for ip in self.particles:
            em_kw = dict(alpha=0.8, ls='-')
            if self.color_pop:
                em_kw['color'] = colors[0] if ip < par_counts[0] else colors[1]
            l_em, = ax.plot(em_x, em_y, **em_kw)
            lines_em.append(l_em)
        self.lines_em = lines_em

        if len(self.particles) <= 20:
            ax.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

        ax.set_ylim(0, 1)
        fig.canvas.draw()
        self.background = fig.canvas.copy_from_bbox(fig.bbox)
        self.title = plt.suptitle('')

    def update(self, slice_=None):
        if slice_ is None:
            slice_ = (0, self.duration_steps, self.decimate)
        slice_ = slice(*slice_[:2])
        assert (slice_.stop - slice_.start) // self.decimate == self.num_points
        emission = self.S.emission[:, slice_]
        dec_shape = (emission.shape[0], emission.shape[1] // self.decimate,
                     self.decimate)
        emission = emission.reshape(*dec_shape).max(axis=-1)

        self.fig.canvas.restore_region(self.background)
        for ip, l_em in zip(self.particles, self.lines_em):
            l_em.set_ydata(emission[ip])
            self.ax.draw_artist(l_em)
        time = slice_.start * self.S.t_step
        self.title.set_text("t = %5.3fs %s" % (time, self.title_suffix))
        self.fig.draw_artist(self.title)
        self.fig.canvas.blit(self.fig.bbox)


class TrackEmPlotterR(ScrollPlotter):
    def __init__(self, S, particles=None, color_pop=True,
                 duration=0.01, decimate=100):
        if particles is None:
            particles = range(S.num_particles)
        self.particles = particles
        self.S = S
        self.position = S.position
        self.color_pop = color_pop
        super().__init__(S.n_samples, duration, S.t_step, decimate)

    def create_figure(self):
        self.fig = plt.figure(figsize=(6, 8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
        ax1 = self.fig.add_subplot(gs[0, 0])
        ax3 = self.fig.add_subplot(gs[1, :])
        self.AX = [ax1, ax3]

    def plot_psf(self):
        psf = self.S.psf.hdata
        cmap = cm.YlGnBu
        cmap.set_under(alpha=0)
        kwargs = dict(interpolation='nearest', cmap=cmap, vmin=1e-1, zorder=1)
        self.AX[0].imshow(psf, extent=(0, 4, -6, 6), **kwargs)

    def init_plot(self):
        fig, AX = self.fig, self.AX
        self.title_suffix = "Total: %.1f s, Visualized: %.2f ms" % (
            self.S.t_step * self.S.n_samples, self.duration * 1e3)

        AX[0].set_xlabel("r (um)")
        AX[0].set_ylabel("z (um)")
        AX[1].set_xlabel("Time (ms)")
        self.plot_psf()

        pal = sns.color_palette()
        colors = pal[1:]
        par_counts = [c[1] for c in self.S.particles.diffusion_coeff_counts]

        em_y = np.zeros(self.num_points)
        em_x = np.arange(self.num_points) * self.S.t_step * self.decimate * 1e3
        lines_rz, lines_em = [], []
        for ip in self.particles:
            color = colors[0] if ip < par_counts[0] else colors[1]
            plot_kwargs = dict(ls='', marker='o', mew=0, ms=2, color=color,
                               alpha=0.5, label='P%d' % ip)
            l_rz, = AX[0].plot([], [], **plot_kwargs)
            em_kw = dict(alpha=0.8, ls='-', color=color)
            l_em, = AX[1].plot(em_x, em_y, **em_kw)
            lines_rz.append(l_rz)
            lines_em.append(l_em)
        self.lines_rz = lines_rz
        self.lines_em = lines_em

        if len(self.particles) <= 20:
            AX[0].legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

        AX[0].set_xlim(0, 4)
        AX[0].set_ylim(-4, 4)
        AX[1].set_ylim(0, 1)
        for ax in AX:
            ax.autoscale(False)
            ax.set_axisbelow(True)
        fig.canvas.draw()
        self.background = fig.canvas.copy_from_bbox(fig.bbox)
        self.title = plt.suptitle('')

    def update(self, slice_=None):
        ax0, ax1 = self.AX
        if slice_ is None:
            slice_ = (0, self.duration_steps, self.decimate)
        slice_ = slice(*slice_)
        assert (slice_.stop - slice_.start) // self.decimate == self.num_points
        pos = self.position[:, :, slice_]
        emission = self.S.emission[:, slice_]

        self.fig.canvas.restore_region(self.background)
        for ip, l_rz, l_em in zip(self.particles,
                                  self.lines_rz,
                                  self.lines_em):
            r, z = pos[ip, 0], pos[ip, 1]
            l_rz.set_data(r * 1e6, z * 1e6)
            l_em.set_ydata(emission[ip])
            ax0.draw_artist(l_rz)
            ax1.draw_artist(l_em)
        time = slice_.start * self.S.t_step
        self.title.set_text("t = %5.1fs %s" % (time, self.title_suffix))
        self.fig.draw_artist(self.title)
        self.fig.canvas.blit(self.fig.bbox)


class TrackEmPlotter(ScrollPlotter):
    def __init__(self, S, particles=None, color_pop=True,
                 duration=0.01, decimate=100):
        if particles is None:
            particles = range(S.num_particles)
        self.particles = particles
        self.S = S
        self.position = S.position
        self.color_pop = color_pop
        super().__init__(S.n_samples, duration, S.t_step, decimate)

    def create_figure(self):
        self.fig = plt.figure(figsize=(11, 8))
        gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])
        ax1 = self.fig.add_subplot(gs[0, 0])
        ax2 = self.fig.add_subplot(gs[0, 1], sharey=ax1)
        plt.setp(ax2.get_yticklabels(), visible=False)
        ax3 = self.fig.add_subplot(gs[1, :])
        self.AX = [ax1, ax2, ax3]

    def plot_psf(self):
        psf = np.concatenate((self.S.psf.hdata[:, :0:-1], self.S.psf.hdata), axis=1)
        cmap = cm.YlGnBu
        cmap.set_under(alpha=0)
        kwargs = dict(interpolation='nearest', cmap=cmap, vmin=1e-1, zorder=1)
        self.AX[1].imshow(psf.T, extent=(-6, 6, -4, 4), **kwargs)

        x = np.concatenate((-self.S.psf.xi[:0:-1], self.S.psf.xi)) * 1e-6
        X, Y = np.meshgrid(x, x)
        R = np.sqrt(X**2 + Y**2)
        psf_xy = self.S.psf.eval_xz(R, 0)
        self.AX[0].imshow(psf_xy, extent=(-4, 4, -4, 4), **kwargs)

    def init_plot(self):
        fig, AX = self.fig, self.AX
        plt.subplots_adjust(left=0.06, right=0.96, top=0.95, bottom=0.09,
                            wspace=0.05)
        self.title_suffix = "Total: %.1f s, Visualized: %.2f ms" % (
            self.S.t_step * self.S.n_samples, self.duration * 1e3)

        AX[1].set_xlabel("z (μm)")
        AX[0].set_xlabel("x (μm)")
        AX[0].set_ylabel("y (μm)")
        AX[2].set_xlabel("Time (ms)")
        self.plot_psf()

        pal = sns.color_palette()
        colors = pal[1:]
        par_counts = [c[1] for c in self.S.particles.diffusion_coeff_counts]

        em_y = np.zeros(self.num_points)
        em_x = np.arange(self.num_points) * self.S.t_step * self.decimate * 1e3
        lines_xy, lines_zy, lines_em = [], [], []
        for ip in self.particles:
            color = colors[0] if ip < par_counts[0] else colors[1]
            plot_kwargs = dict(ls='', marker='o', mew=0, ms=2, color=color,
                               alpha=0.5, label='P%d' % ip)
            l_xy, = AX[0].plot([], [], **plot_kwargs)
            l_zy, = AX[1].plot([], [], **plot_kwargs)
            em_kw = dict(alpha=0.8, ls='-', color=color)
            l_em, = AX[2].plot(em_x, em_y, **em_kw)
            lines_xy.append(l_xy)
            lines_zy.append(l_zy)
            lines_em.append(l_em)
        self.lines_xy = lines_xy
        self.lines_zy = lines_zy
        self.lines_em = lines_em

        if len(self.particles) <= 20:
            AX[1].legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

        AX[0].set_xlim(-4, 4)
        AX[0].set_ylim(-4, 4)
        AX[1].set_xlim(-4, 4)
        AX[2].set_ylim(0, 1)
        for ax in AX:
            ax.autoscale(False)
            ax.set_axisbelow(True)
        fig.canvas.draw()
        self.background = fig.canvas.copy_from_bbox(fig.bbox)
        self.title = plt.suptitle('')

    def update(self, slice_=None):
        if slice_ is None:
            slice_ = (0, self.duration_steps, self.decimate)
        slice_ = slice(*slice_)
        assert (slice_.stop - slice_.start) // self.decimate == self.num_points
        pos = self.position[:, :, slice_]
        emission = self.S.emission[:, slice_]

        self.fig.canvas.restore_region(self.background)
        for ip, l_xy, l_zy, l_em in zip(self.particles,
                                        self.lines_xy, self.lines_zy,
                                        self.lines_em):
            x, y, z = pos[ip, 0], pos[ip, 1], pos[ip, 2]
            l_xy.set_data(x * 1e6, y * 1e6)
            l_zy.set_data(z * 1e6, y * 1e6)
            l_em.set_ydata(emission[ip])
            self.AX[0].draw_artist(l_xy)
            self.AX[1].draw_artist(l_zy)
            self.AX[2].draw_artist(l_em)
        time = slice_.start * self.S.t_step
        self.title.set_text("t = %5.1fs %s" % (time, self.title_suffix))
        self.fig.draw_artist(self.title)
        self.fig.canvas.blit(self.fig.bbox)
