from enthought.traits.api import Enum, Int
from enthought.chaco.api import AbstractOverlay
from enthought.enable.api import ColorTrait

class Line(AbstractOverlay):
    orientation = Enum("h", "v")
    color = ColorTrait("green")
    width = Int(1)
    def overlay(self, component, gc, *args, **kw):
        if component is None:
            return
        gc.save_state()
        gc.set_stroke_color(self.color_)
        gc.set_line_width(self.width)
        if self.orientation == "h":
            mid = component.y + component.height/2
            gc.move_to(component.x, mid)
            gc.line_to(component.x2, mid)
        else:
            mid = component.x + component.width/2
            gc.move_to(mid, component.y)
            gc.line_to(mid, component.y2)
        gc.stroke_path()
        gc.restore_state()
