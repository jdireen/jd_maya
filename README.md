# jd_maya
A small collection (currently just 2, real small...) of Maya DraggerContext tools.

## SlideVertexWeightsTool
Assign skin weights to a specified influence.

With vertex components selected (soft selections work too), activate the tool with the following code:
```
import jd_DraggerContexts

tool = jd_DraggerContexts.SlideVertexWeightsTool( multiplier=0.01, minValue=0.0, maxValue=1.0, incriment=0.01, nearInfThreshold=0.0005)
tool.setTool()
```
Your cursor will change to a crosshair. Then click-and-drag on the joint you want to slide the weights too. If multiple joints are near, a dialog will pop-up requesting a selection.

## PlungeJointTool
Quickly place new joints in the center of a mesh volume.

Activate the tool with:
```
import jd_DraggerContexts

tool = jd_DraggerContexts.PlungeJointTool(radius=0.1)
tool.setTool()
```
Once active, your cursor will change to a crosshair. Now wherever you click over a mesh object, a joint will be placed halfway between the front to the back from the viewports camera view. If you hold control while clicking, the new joint will be parented to the previously created joint. After placing a joint, you can middle-mouse-drag over anywhere in the viewport to adjust the depth, from front to back of the initial placement of the joint.

# Requirements
Requires the awesome apiundo by mottosso: https://github.com/mottosso/apiundo/tree/master - Copyright (c) 2024, Marcus Ottosson

Note that my initial DraggerContext class was inspired / modified from Morgan Loomis' found here: https://github.com/TinyPHX/MayaPythonTools/blob/master/ml_utilities.py
