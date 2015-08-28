# 
# Implements the modified plot window for scrolling timetraces plots
#

from PySide import QtGui, QtCore

class RangeToolQT(object):
    def __init__(self, fig):
        # Setup data range variables for scrolling
        self.fig = fig
        self.xmin, self.xmax = fig.axes[0].get_xlim()
        self.ymin, self.ymax = fig.axes[0].get_ylim()
        
        # Some handy shortcuts
        self.ax = self.fig.axes[0]
        self.draw = self.fig.canvas.draw

        # Create the spinboxes for setting the range
        self.set_spinbox(fig.canvas.toolbar)
        self.draw()

    def set_spinbox(self, parent):
        self.xmin_sb = QtGui.QDoubleSpinBox()
        self.xmax_sb = QtGui.QDoubleSpinBox()
        self.ymin_sb = QtGui.QDoubleSpinBox()
        self.ymax_sb = QtGui.QDoubleSpinBox()
        sb_list = [self.xmin_sb,self.xmax_sb,self.ymin_sb,self.ymax_sb]
        values = [self.xmin, self.xmax, self.ymin, self.ymax]
        for spinbox, val in zip(sb_list, values):
            spinbox.setDecimals(4)
            spinbox.setRange(-1e9,1e9)
            spinbox.setValue(val)   # set the initial width
            spinbox.valueChanged.connect(self.range_changed)
            parent.addWidget(spinbox)
            spinbox.setMinimumWidth(100)
            #spinbox.setMaximumWidth(100)

    def range_changed(self, *args):
        self.ax.set_xlim(self.xmin_sb.value(), self.xmax_sb.value())
        self.ax.set_ylim(self.ymin_sb.value(), self.ymax_sb.value())
        self.draw()

class mToolQT(object):
    def __init__(self, fig, plot_fun, *args, **kwargs):
        if 'bins' not in kwargs: kwargs.update(bins=r_[:10:0.02])
        if 't_max' not in kwargs: kwargs.update(t_max=-1)
       
        # Save some variables
        self.fig = fig
        self.AX = kwargs['AX']
        self.plot_fun = plot_fun
        self.f_args = args
        self.f_kwargs = kwargs
        self.time_max = args[0].time_max()
        
        # Some handy shortcut
        self.draw = self.fig.canvas.draw

        # Create the spinboxes for setting the range
        mlabel = QtGui.QLabel()
        mlabel.setText("m:")
        self.m_spinbox = QtGui.QSpinBox()
        self.m_spinbox.setRange(2,500)
        self.m_spinbox.setKeyboardTracking(False)
        self.m_spinbox.setValue(self.f_kwargs['m'])
        self.m_spinbox.valueChanged.connect(self.par_changed)
        
        blabel = QtGui.QLabel()
        blabel.setText("bin:")
        self.bin_spinbox = QtGui.QDoubleSpinBox()
        self.bin_spinbox.setRange(1e-6,1e6)
        self.bin_spinbox.setDecimals(3)
        self.bin_spinbox.setKeyboardTracking(False)
        bins = self.f_kwargs['bins']
        self.bin_spinbox.setValue(bins[1]-bins[0])
        self.bin_spinbox.valueChanged.connect(self.par_changed)
        
        tlabel = QtGui.QLabel()
        tlabel.setText("t max (s):")
        self.tmax_spinbox = QtGui.QDoubleSpinBox()
        self.tmax_spinbox.setRange(0,3600)
        self.tmax_spinbox.setKeyboardTracking(False)
        if kwargs['t_max'] <= 0 or kwargs['t_max'] > self.time_max:
            kwargs['t_max'] = self.time_max
        self.tmax_spinbox.setValue(kwargs['t_max'])
        self.tmax_spinbox.valueChanged.connect(self.par_changed)
        
        addWidget = fig.canvas.toolbar.addWidget
        addWidget(mlabel); addWidget(self.m_spinbox)
        addWidget(blabel); addWidget(self.bin_spinbox)
        addWidget(tlabel); addWidget(self.tmax_spinbox)
        self.draw()
        self.save_params()

    def save_params(self):
        self.params = dict(m = self.m_spinbox.value(),
                bin_w = self.bin_spinbox.value(),
                t_max = self.tmax_spinbox.value())

    def par_changed(self, *args):
        #for ax in self.AX.ravel(): 
        #    for i in range(len(ax.lines)): ax.lines.pop()
        #    #ax.clear()
        #print ".",
        old = self.params
        new = dict(m = self.m_spinbox.value(),
                bin_w = self.bin_spinbox.value(),
                t_max = self.tmax_spinbox.value())
        if array([new[p] == old[p] for p in new]).all():
            print "all same"
            return
        #print new
        
        if new['t_max'] > self.time_max:
            self.tmax_spinbox.setValue(self.time_max)
            self.save_params()
            print "t_max too large"
            return

        bins = self.f_kwargs['bins']
        if new['bin_w'] <= 0 or new['bin_w'] >= 0.5*bins.max():
            self.bin_spinbox.setValue(old['bin_w'])
            self.save_params()
            print "bins out of range"
            return
        bins = arange(bins.min(),bins.max()+new['bin_w'],new['bin_w'])
        
        self.save_params()
        self.f_kwargs.update(m=new['m'], t_max=new['t_max'], bins=bins)
        self.plot_fun(*self.f_args, **self.f_kwargs)
        self.draw()
        
class ScrollingToolQT(object):
    def __init__(self, fig, page_step=1):
        """Make a scrolling x axis on figure `fig`.
        `page_step` is the multiplier for page-step scrolling.
        """
        # Setup data range variables for scrolling
        self.fig = fig
        self.page_step = page_step # configurable for page step
        self.xmin, self.xmax = fig.axes[0].get_xlim()
        self.width = min(1, self.xmax-self.xmin) # axis units
        self.pos = 0   # axis units
        self.scale = 1e3 # conversion between scrolling units and axis units
        
        # Some handy shortcuts
        self.ax = self.fig.axes[0]
        self.draw = self.fig.canvas.draw
        #self.draw_idle = self.fig.canvas.draw_idle

        # Retrive the QMainWindow used by current figure and add a toolbar
        # to host the new widgets
        QMainWin = fig.canvas.parent()
        toolbar = QtGui.QToolBar(QMainWin)
        QMainWin.addToolBar(QtCore.Qt.BottomToolBarArea, toolbar)
        
        # Create the slider and spinbox for x-axis scrolling in toolbar
        self.set_slider(toolbar)
        self.set_spinbox(toolbar)
        
        # Set the initial xlimits coherently with values in slider and spinbox
        self.ax.set_xlim(self.pos,self.pos+self.width)
        self.draw()

    def set_slider(self, parent):
        self.slider = QtGui.QSlider(QtCore.Qt.Horizontal, parent=parent)
        self.slider.setTickPosition(QtGui.QSlider.TicksAbove)
        self.slider.setTickInterval((self.xmax-self.xmin)/10.*self.scale)
        self.slider.setMinimum(self.xmin*self.scale)
        if (self.xmax-self.width) > 0:
            self.slider.setMaximum((self.xmax-self.width)*self.scale)
        else:
            self.slider.setMaximum(self.xmax*self.scale)
        self.slider.setSingleStep(self.width*self.scale/4.)
        self.slider.setPageStep(self.page_step*self.width*self.scale)
        self.slider.setValue(self.pos*self.scale) # set the initial position
        self.slider.valueChanged.connect(self.xpos_changed)
        parent.addWidget(self.slider) 

    def set_spinbox(self, parent):
        self.spinb = QtGui.QDoubleSpinBox(parent=parent)
        self.spinb.setDecimals(4)
        self.spinb.setRange(0.0001,3600.)
        self.spinb.setSuffix(" s")
        self.spinb.setValue(self.width)   # set the initial width
        self.spinb.valueChanged.connect(self.xwidth_changed)
        parent.addWidget(self.spinb)
        
    def xpos_changed(self, pos):
        #pprint("Position (in scroll units) %f\n" %pos)
        pos /= self.scale
        self.ax.set_xlim(pos, pos+self.width)
        self.draw()

    def xwidth_changed(self, width):
        #pprint("Width (axis units) %f\n" % step)
        if width <= 0: return
        self.width = width
        self.slider.setSingleStep(self.width*self.scale/4.)
        self.slider.setPageStep(self.page_step*self.width*self.scale)
        old_xlim = self.ax.get_xlim()
        self.xpos_changed(old_xlim[0]*self.scale)



