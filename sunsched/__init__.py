"""SunSched: deciding *when* to run inference on a solar camera trap, not only how much.

On real camera-trap data most animal events arrive at night, when solar harvest
is zero, while most energy arrives at midday, when there is little to classify.
The battery that bridges the two ages mainly by sitting charged. SunSched
therefore triages every capture cheaply when it happens, defers the expensive
classification of uncertain frames into the daytime surplus, and manages the
battery as costly storage with a risk-controlled charge target.
"""
__version__ = "2.0.0"
