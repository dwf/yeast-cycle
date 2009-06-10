from enthought.traits.api import TraitRange, TraitFactory, HasTraits, Trait

#-------------------------------------------------------------------------------
#  'DynamicTraitRange' class:
#-------------------------------------------------------------------------------

class DynamicTraitRange ( TraitRange ):

     def __init__ ( self, low = None, high = None ):
         super( DynamicTraitRange, self ).__init__( low, high )
         del self.fast_validate  # Don't use the C-level validator!

#-------------------------------------------------------------------------------
#  Factory function for creating dynamic range traits:
#-------------------------------------------------------------------------------

def DynamicRange ( value, range = None, **metadata ):
     if range is None:
         range = value
         value = range._low
     return Trait( value, range, **metadata )

DynamicRange = TraitFactory( DynamicRange )

#-------------------------------------------------------------------------------
#  Test Case/Example:
#-------------------------------------------------------------------------------

dr = DynamicTraitRange( 0, 10 )

class foo ( HasTraits ):
     bar = DynamicRange( dr )
     baz = DynamicRange( 5, dr )

zz = foo()
zz.print_traits()
zz.bar = 10
zz.print_traits()
try:
     zz.bar = 11
except:
     import traceback
     traceback.print_exc()

dr._high = 20

zz.bar  = 11
zz.print_traits()
try:
     zz.baz = 21
except:
     import traceback
     traceback.print_exc()

