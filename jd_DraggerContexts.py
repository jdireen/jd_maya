#!/usr/bin/env python

#----------------------------------------------------------------------------#
#------------------------------------------------------------------ HEADER --#
"""
------------------------------------------------------------------------------

:Authors:
    - James Direen
    
    
:License:
    Copyright 2024 James Direen
    MIT License
    https://opensource.org/license/mit


:Description:
    Maya DraggerContext Tools

"""

#----------------------------------------------------------------------------#
#----------------------------------------------------------------- IMPORTS --#

from maya import cmds

import maya.OpenMaya as om
import maya.OpenMayaUI as omui

from maya.api import OpenMaya as OpenMaya2
from maya.api import OpenMayaAnim as OpenMayaAnim2

import apiundo

#----------------------------------------------------------------------------#
#--------------------------------------------------------------- FUNCTIONS --#

def asList(arg):

    if arg is None: return []

    if isinstance(arg, str):
        return [arg]
    else:
        try:
            return list(arg)
        except TypeError:
            return [arg]


def getNodes(nodes=None, types=None, long=False, absoluteName=False):

    if not nodes:
        nodes = cmds.ls(sl=True, long=long, absoluteName=absoluteName)
        if not nodes: raise ValueError('No nodes were provided.')

    if isinstance(nodes, str):
        if '*' in nodes:
            nodes = cmds.ls(nodes, long=long, absoluteName=absoluteName)

    nodes = asList(nodes)

    if types:
        types = set(asList(types))
        nodes = filterNodes(nodes, types=types)

    return nodes


def filterNodes(nodes, types=None):
    
    result = nodes[:]
    if not types: return result
    
    types = asList(types)
    
    validNodes = []
    for n in result:
        for typ in types:
            if not cmds.objectType(n, isAType=typ):
                continue
            validNodes.append(n)
            break

    result = validNodes

    return result


def getMObject(node):

    selectionList = OpenMaya2.MSelectionList()
        
    try:
        selectionList.add(node)
    except RuntimeError:
        raise ValueError("No object matches name '{}'.".format(node))

    mObj = selectionList.getDependNode(0)

    return mObj


def findRelatedDeformers(nodes=None, types=None, checkOutput=False):

    nodes = getNodes(nodes)

    if not types:
        types = "geometryFilter"

    matches = []
    for node in nodes:
        if getNodes(node, types=types):
            matches.append(node)
            continue

        elif cmds.objectType(node, isAType="shape"):
            shapes = [node]
        else:
            shapes = cmds.listRelatives(node, shapes=True, path=True)
            if not shapes:
                continue

        for shape in shapes:
            if checkOutput:
                temp_def = cmds.listConnections(shape, s=0, d=1, scn=True)
                found = getNodes(temp_def, types=types)
                if found:
                    matches.extend(found)

                continue

            history = getNodes(cmds.listHistory(shape), types=types)
            for item in history:
                result = cmds.deformer(item, query=True, geometry=True)
                if result is None:
                    cmds.warning(
                        "Deformer {} not associated with any geometry. "
                        "".format(item)
                    )
                    continue

                if shape not in cmds.deformer(item, query=True, geometry=True):
                    continue
                matches.append(item)

    return matches


def getSkinClusterInfluences(skinCluster: OpenMaya2.MObject) -> list:

    fnSkinCluster = OpenMayaAnim2.MFnSkinCluster(skinCluster)
    influenceObjs = fnSkinCluster.influenceObjects()

    return [OpenMaya2.MFnDagNode(influenceObjs[i]).name() 
            for i in range(len(influenceObjs))]


def getSkinWeightsFromSelectedComponents(skinCluster: OpenMaya2.MObject):

    fnSkinCluster = OpenMayaAnim2.MFnSkinCluster(skinCluster)
    
    selection = OpenMaya2.MGlobal.getRichSelection().getSelection()
    dagPath, components = selection.getComponent(0)
    fnComp = OpenMaya2.MFnSingleIndexedComponent(components)
    
    softSelectionWeights = []
    if fnComp.hasWeights:
        selectedIdxs = fnComp.getElements()
        for i in range(len(selectedIdxs)): 
            softSelectionWeights.append(fnComp.weight(i).influence)
    
    weights, infNum = fnSkinCluster.getWeights(dagPath, components)
    return weights, infNum, softSelectionWeights


def setSkinWeightsToSelectedComponents(skinCluster: OpenMaya2.MObject, 
                          weights: OpenMaya2.MDoubleArray, 
                          influences: OpenMaya2.MIntArray):

    fnSkinCluster = OpenMayaAnim2.MFnSkinCluster(skinCluster)
   
    selection = OpenMaya2.MGlobal.getRichSelection().getSelection()
    dagPath, components = selection.getComponent(0)
   
    fnSkinCluster.setWeights(
                             dagPath, 
                             components, 
                             influences, 
                             weights, 
                             True, 
                             False
                             )

#----------------------------------------------------------------------------#
#----------------------------------------------------------------- CLASSES --#

class DraggerContext(object):
    """
    Inspired by: 
    https://github.com/TinyPHX/MayaPythonTools/blob/master/ml_utilities.py
    """
    CTX_NAME = 'myDraggerCtx'

    def __init__(self,
                name = 'myDraggerCTX',
                title = 'Dragger',
                defaultValue=0,
                minValue=None,
                maxValue=None,
                multiplier=0.01,
                cursor='crossHair',
                space='screen',
                projection='viewPlane'
                ):
        
        self.button = None # 1:left | 2:middle | 3:right
        self.modifier= None # ctrl | shift | alt | none
        
        self.multiplier = multiplier
        self.defaultValue = defaultValue
        self.minValue = minValue
        self.maxValue = maxValue
        self.space = space
        self.projection = projection
        
        self.anchorPoint = 0.0, 0.0, 0.0
        
        self.CTX_NAME = name
        
        if cmds.draggerContext(self.CTX_NAME, exists=True):
            cmds.deleteUI(self.CTX_NAME)
            
        self.CTX_NAME = cmds.draggerContext(self.CTX_NAME,
                                            pressCommand=self._onPress, 
                                            dragCommand=self._onDrag,
                                            releaseCommand=self._onRelease,
                                            cursor=cursor,
                                            drawString=title,
                                            undoMode='all',
                                            space=self.space
                                            )
                                                    
    
    def _onPress(self):
        self.anchorPoint = cmds.draggerContext(self.CTX_NAME, 
                                               query=True, 
                                               anchorPoint=True)
        self.button = cmds.draggerContext(self.CTX_NAME, 
                                          query=True, 
                                          button=True)
        self.modifier = cmds.draggerContext(self.CTX_NAME, 
                                            query=True, 
                                            modifier=True)
        
        # This turns off the undo queue until we're done dragging, so we can undo it.
        cmds.undoInfo(openChunk=True)
        self.onPress()
        
    def onPress(self):
        pass
        
    def _onDrag(self):
        self.dragPoint = cmds.draggerContext(self.CTX_NAME, 
                                             query=True, 
                                             dragPoint=True)
        
        self.x = ((self.dragPoint[0] - self.anchorPoint[0]) * self.multiplier) + self.defaultValue
        self.y = ((self.dragPoint[1] - self.anchorPoint[1]) * self.multiplier) + self.defaultValue
        
        if self.minValue is not None and self.x < self.minValue:
            self.x = self.minValue
        if self.maxValue is not None and self.x > self.maxValue:
            self.x = self.maxValue
        if self.minValue is not None and self.y < self.minValue:
            self.y = self.minValue
        if self.maxValue is not None and self.y > self.maxValue:
            self.y = self.maxValue            
        self.onDrag()
        
        cmds.refresh()
        
    def onDrag(self):
        pass
    
    def _onRelease(self):
        # close undo chunk
        self.onRelease()
        cmds.undoInfo(closeChunk=True)
        
    def onRelease(self):
        pass
    
    def drawString(self, message):
        '''
        Creates a string message at the position of the pointer.
        Does this work? Maybe something needs too be enabled in the UI...?
        '''
        cmds.draggerContext(self.CTX_NAME, edit=True, drawString=message)
    
    #no drag right, because that is monopolized by the right click menu
    #no alt drag, because that is used for the camera
    
    def setTool(self):
        cmds.setToolTo(self.CTX_NAME)
        
        
class SlideVertexWeightsTool(DraggerContext):
    
    
    def __init__(self, nodes=None, 
                 mappedJoints=None, 
                 incriment=0.01, 
                 nearInfThreshold=0.0005,
                 **kwargs):
        
        super().__init__(name='SlideVertexWeightsToolCTX', **kwargs)
        
        nodes = getNodes(nodes)
        
        assert '[' in nodes[0], 'must have verticies selected'
        
        self.selVerts = cmds.ls(nodes, flatten=True)
        selNode = nodes[0].split('.')[0]
        
        self.incriment = incriment
        self.nearInfThreshold = nearInfThreshold
        self.closestJoint = None
        self.closestValue = 0.0
        self.closestJointIdx = 0
        self.closestJointMap = {}
        self.slid = False
        
        sc = findRelatedDeformers(selNode, 'skinCluster')
        
        assert sc, f'No skinCluster found on selected object: {selNode}'
        if len(sc) > 1:
            raise ValueError('More than one skinCluster found on selected')
        
        self.skinCluster = sc[0]
        self.skinClusterMObject = getMObject(sc[0])
        
        joints = getSkinClusterInfluences(self.skinClusterMObject)
        self.influenceNames = joints
        
        if not mappedJoints:
            jntPoses = [cmds.xform(x, q=True, ws=True, t=True) for x in joints]
            mappedJoints = {jnt:point for (jnt,point) in zip(joints, jntPoses)}
            
        self.mappedJoints = mappedJoints
        
        # updated when getStartingWeights is called
        self.startingWeights = None
        self.numberOfInfluences = None
        self.numberOfWeights = None
        self.influenceIndices = None 
        self.significantWeights = None
        self.softSelectionWeights = None
        
        self.getStartingWeights()
        self.slidWeights = OpenMaya2.MDoubleArray(self.startingWeights)
        
        
    def onPress(self):
        if self.closestJoint is not None: return # should only hit this when a confirmDialog was prompted
        
        cmds.scriptEditorInfo(suppressWarnings=True)
        
        vpX, vpY, _ = self.anchorPoint
        
        pos = om.MPoint()
        ray = om.MVector()
        omui.M3dView().active3dView().viewToWorld(int(vpX), int(vpY), pos, ray)
        ray.normalize()
        
        lastResult = 0.0
        for jnt, point in self.mappedJoints.items():
    
            testVector = om.MPoint(*point) - pos
            testVector.normalize()
            
            result =  testVector * ray

            if result > lastResult:
                lastResult = result
                self.closestJoint = jnt
                self.closestValue = result
                self.closestJointIdx = self.influenceNames.index(jnt)
                
            self.closestJointMap[result] = jnt
        
        # if joints near enough, popup choice dialog for user selection
        test = [x for x in self.closestJointMap.keys() if x > 0.99]
        closeJointsKeys = [x for x in self.closestJointMap.keys() 
                           if self.closestValue - x < self.nearInfThreshold]
        if len(closeJointsKeys) > 1:
            closeJoints = [self.closestJointMap[x] for x in closeJointsKeys]
            choice = cmds.confirmDialog(
                    title='Multiple Influences Near',
                    message='Choose Influence',
                    button=closeJoints,
                    cancelButton='Cancel',
                    dismissString='Cancel'
                )
            self.closestJoint = choice
                
        cmds.inViewMessage( 
                    amg=f'Target Joint Set: <hl>{self.closestJoint}</hl>.', 
                    pos='topCenter', fade=True )
                
    def onDrag(self):
        incriment = self.incriment
        
        if self.modifier == 'ctrl':
            incriment *= 0.1
        elif self.modifier == 'shift':
            incriment *= 10
        
        self.slideWeights()
        self.slid = True
        
    def onRelease(self):
        if not self.slid: return
        cmds.scriptEditorInfo(suppressWarnings=False)
        cmds.SelectTool()
        
    def getStartingWeights(self):
        weights, infNum, softSelectionWeights = getSkinWeightsFromSelectedComponents(self.skinClusterMObject)
        self.startingWeights = weights
        self.numberOfInfluences = infNum
        self.numberOfWeights = len(weights)
        
        if not softSelectionWeights:
            softSelectionWeights = [1.0 for x in range(self.numberOfWeights)]
        self.softSelectionWeights = softSelectionWeights
        
        infList = [x for x in range(infNum)] 
        self.influenceIndices = OpenMaya2.MIntArray(infList)
        
        self.significantWeights = [x != 0.0 for x in self.startingWeights]
        
    def updateWeights(self):
        setSkinWeightsToSelectedComponents(self.skinClusterMObject, 
                                           self.slidWeights, 
                                           self.influenceIndices)
        
    def slideWeights(self):
        
        k = 0
        for i in range(0, self.numberOfWeights, self.numberOfInfluences):
            for j in range(self.numberOfInfluences):
                idx = i + j
                if j == self.closestJointIdx:
                    newWeight = self.startingWeights[idx] + (((1.0 - self.startingWeights[idx]) * self.x) * self.softSelectionWeights[k])
                    self.slidWeights[idx] = newWeight
                else:
                    if not self.significantWeights[idx]: continue
                    newWeight = self.startingWeights[idx] - ((self.startingWeights[idx] * self.x) * self.softSelectionWeights[k])
                    self.slidWeights[idx] = newWeight
            k += 1
                
        self.updateWeights()
        apiundo.commit(self.undoIt, self.redoIt)
        
    def undoIt(self):
        setSkinWeightsToSelectedComponents(self.skinClusterMObject, 
                                           self.startingWeights, 
                                           self.influenceIndices)
    
    def redoIt(self):
        self.updateWeights()
        
        
class PlungeJointTool(DraggerContext):
    
    
    def __init__(self, radius=1.0, **kwargs):
        
        self.jointRadius = radius
        
        super().__init__(name='PlungeJointToolCTX', **kwargs)
        
        
        self.frontHitpoint = om.MFloatPoint()
        self.backHitpoint = om.MFloatPoint()
        
        self.lastCreatedJoint = None
        
        
    def onPress(self):
        
        if self.button != 1: return
        
        vpX, vpY, _ = self.anchorPoint
        
        frontPoint = om.MPoint()
        backPoint = om.MPoint()
        omui.M3dView().active3dView().viewToWorld(int(vpX), 
                                                  int(vpY), 
                                                  frontPoint, 
                                                  backPoint)
        
        forwardVector = backPoint - frontPoint
        forwardVector.normalize()
        backVector = forwardVector * -1
        
        pointFound = False
        for mesh in cmds.ls(type='mesh'):
            selectionList = om.MSelectionList()
            selectionList.add(mesh)
            dagPath = om.MDagPath()
            selectionList.getDagPath(0, dagPath)
            fnMesh = om.MFnMesh(dagPath)
            frontIntersect = fnMesh.closestIntersection(
                om.MFloatPoint(frontPoint),
                om.MFloatVector(forwardVector),
                None,
                None,
                False,
                om.MSpace.kWorld,
                99999,
                False,
                None,
                self.frontHitpoint,
                None,
                None,
                None,
                None,
                None)
            
            backIntersect = fnMesh.closestIntersection(
                om.MFloatPoint(backPoint),
                om.MFloatVector(backVector),
                None,
                None,
                False,
                om.MSpace.kWorld,
                99999,
                False,
                None,
                self.backHitpoint,
                None,
                None,
                None,
                None,
                None)
            
            if frontIntersect and backIntersect:
                pointFound = True
                break
                
        if pointFound:
            a = self.frontHitpoint
            b = self.backHitpoint
            
            jntPnt = a + (b - a) / 2
            
            if self.modifier == 'ctrl' and self.lastCreatedJoint is not None:
                cmds.select(self.lastCreatedJoint, r=True)
            else:
                cmds.select(clear=True)
                
            jnt = cmds.joint(p=(jntPnt.x, jntPnt.y, jntPnt.z), 
                             radius=self.jointRadius)
            
            self.lastCreatedJoint = jnt
        
                
    def onDrag(self):
        if self.button == 2:
            a = self.frontHitpoint
            b = self.backHitpoint
            
            m = 0.5 + (self.x * -1)
            m = 0 if m < 0 else m
            m = 1 if m > 1 else m
            
            jntPnt = a + (b - a) * m
            
            cmds.xform(self.lastCreatedJoint, ws=True,
                       t=(jntPnt.x, jntPnt.y, jntPnt.z))
            
        
    def onRelease(self):
        pass