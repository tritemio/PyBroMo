#
# Implements the modified plot window for scrolling timetraces plots
#
import sys
from PyQt4 import QtGui, QtCore

def pprint(s):
    """Print immediately, even if inside a busy loop."""
    sys.stdout.write(s)
    sys.stdout.flush()

def Toolbar(*widget_list):
    tool = QtGui.QToolBar()
    for widget in widget_list: tool.addWidget(widget)
    return tool

def Slider(parent, slot, xmin=0, xmax=100, ticks_pos='Above'):
    sl = QtGui.QSlider(QtCore.Qt.Horizontal, parent=parent)
    sl.setTickPosition(getattr(QtGui.QSlider, 'Ticks'+ticks_pos))
    sl.setTickInterval((xmax-xmin)/10.)
    sl.setMinimum(xmin); sl.setMaximum(xmax)
    sl.valueChanged.connect(slot)
    return sl

def Spinbox(parent, slot, xwidth):
    spin = QtGui.QDoubleSpinBox(parent=parent)
    spin.setRange(0.001,3600.)
    spin.setSuffix(" s")
    spin.setValue(xwidth)
    spin.valueChanged.connect(slot)
    return spin

class ScrollingToolQT(object):
    def __init__(self, fig):
        self.fig = fig
        xmin, xmax = fig.axes[0].get_xlim()
        self.max_range = xmax-xmin
        self.step = 1
        self.pos = 0

        QMainWin = fig.canvas.parent()
        toolbar = QtGui.QToolBar(QMainWin)
        QMainWin.addToolBar(QtCore.Qt.BottomToolBarArea, toolbar)

        self.xpos_slider = Slider(toolbar, self.xpos_changed, xmin*1e3,xmax*1e3)
        self.xpos_slider.setSingleStep(self.step*1e3/5.)
        self.xpos_slider.setPageStep(self.step*1e3)
        self.xwidth_spinbox = Spinbox(toolbar, self.xwidth_changed, xmax-xmin)
        self.xwidth_spinbox.setValue(self.step)

        toolbar.addWidget(self.xpos_slider)
        toolbar.addWidget(self.xwidth_spinbox)

        self.fig.axes[0].set_xlim(self.pos,self.pos+self.step)
        self.fig.canvas.draw()

    def xpos_changed(self, pos):
        #pprint("Position %f\n" %pos)
        self.pos = pos
        self.fig.axes[0].set_xlim(self.pos*1e-3,self.pos*1e-3+self.step)
        self.fig.canvas.draw()

    def xwidth_changed(self, step):
        #pprint("Step %f\n" %step)
        if step <=0:
            step = self.max_range
            self.xwidth_spinbox.setSliderPosition(0)
        self.step = step
        self.xpos_slider.setSingleStep(self.step*1e3/5.)
        self.xpos_slider.setPageStep(self.step*1e3)
        self.fig.axes[0].set_xlim(self.pos,self.pos+self.step)
        self.fig.canvas.draw()

