# -*- coding: utf-8 -*-

import openseespy.opensees as ops
import numpy as np
import matplotlib.pyplot as plt
import sys

nSpaces = 110
OpenSeesHeader = {"header_00": " ",
                  "header_01": "OpenSees ── Open System For Earthquake Engineering Simulation",
                  "header_02": "Pacific Earthquake Engineering Research Center (PEER)",
                  "header_03": "OpenSees " + ops.version() + " 64-Bit",
                  "header_04": "Python " + sys.version,
                  "header_05": " ",
                  "header_06": "(c) Copyright 1999-2024 The Regents of the University of California",
                  "header_07": "All Rights Reserved",
                  "header_08": "(Copyright and Disclaimer @ http://www.berkeley.edu/OpenSees/copyright.html)",
                  "header_09": " ",
                  "header_10": "McKenna, F., Fenves, G. L, and Scott, M. H. (2000)",
                  "header_11": "Open System for Earthquake Engineering Simulation. University of California, Berkeley",
                  "header_12": "https://opensees.berkeley.edu/",
                  "header_13": " ",
                  }
    
def title(title="Title Example!", nSpaces=nSpaces):
    header = (nSpaces) * "─"
    print("┌" + header.center((nSpaces), " ") + "┐")
    print("│" + title.center((nSpaces), " ") +  "│")
    print("└" + header.center((nSpaces), " ") + "┘")
    
        
def OPSheader(title, nSpaces=nSpaces):
    print("┌── OpenSees Information " + (nSpaces-24) * "─" + "┐")
    for i in title.keys():
        print("│" + title[i].center(nSpaces, " ") +  "│")
    print("└" + (nSpaces)*"─" + "┘")
    print(" ")
    
OPSheader(title=OpenSeesHeader, nSpaces=nSpaces)


# ┌───────────────────────────────────────────────────────────────────────────────┐
# │                          Defining some functions                              │
# └───────────────────────────────────────────────────────────────────────────────┘
def plot_fig(figure_size=[8, 6], font_size=11, use_latex=False):
    """
    Function for plotting LaTeX figures (publication ready - Computer Modern)
    """
    fig_width_mm = figure_size[0] * 10
    fig_height_mm = figure_size[1] * 10

    fig_width = (fig_width_mm * 2.845275696) * (1.0 / 72.27)  # width in inches
    fig_height = (fig_height_mm * 2.845275696) * (1.0 / 72.27)  # height in inches
    fig_size = [fig_width, fig_height]
    fig = plt.figure(figsize=fig_size)
    ax = plt.axes()

    plt.rcParams["font.size"] = font_size

    if use_latex == True:
        plt.rcParams["text.usetex"] = True
        plt.rcParams["font.family"] = "STIXGeneral"
        plt.rcParams["mathtext.fontset"] = "stix"
    
    plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['figure.titlesize'] = plt.rcParams['font.size']
    plt.rcParams['axes.titlesize'] = plt.rcParams['font.size']
    plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
    plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['axes.axisbelow'] = True

    plt.grid(True, color="silver", linestyle="solid",
             linewidth=1.0, alpha=0.5)
    plt.rc("axes", axisbelow=True)
    plt.tight_layout(pad=0.1)  # 0.1
    plt.tick_params(direction="in", length=5, colors="k", width=0.75)

    return ax, fig


# ┌───────────────────────────────────────────────────────────────────────────────┐
# │                             Units and Constants                               │
# └───────────────────────────────────────────────────────────────────────────────┘
# Basic Units [m], [kN], [rad], [kPa], [tonne], [m/s²], [m/s], [sec], [Hz]
m, kN, sec = 1.0, 1.0, 1.0  # meter for length, kilonewton for force, second for time

# Angle
rad = 1.0
deg = np.pi/180.0*rad

# Length, Area, Volume, Second moment of area
m2, m3, m4 = m**2, m**3, m**4
cm, cm2, cm3, cm4 = m*1E-2, m*1E-4, m*1E-6, m*1E-8
mm, mm2, mm3, mm4 = m*1E-3, m*1E-6, m*1E-9, m*1E-12
inch = 0.0254*m
ft = 0.3048*m

# Force
N = kN*1E-3
g = 9.80665*m/(sec**2)

# Mass
kg = N*sec**2/m
tonne = kg*1E3
lbs = 0.45359237*kg  # pound (lb, lbs)
kip = 453.59237*kg  # kip or kip-force, or kilopound (kip, klb, kipf)

# Pressure
Pa, kPa, MPa, GPa = N/m**2, 1E3*N/m**2, 1E6*N/m**2, 1E9*N/m**2
pcf = lbs/(ft**3)  # pound per cubic foot (lb/ft³)
ksi = 6894.7572931783*kPa # kip/(inch**2)  # kip per square inch (kip/in², ksi)
psi = ksi/1E3  # pound-force per square inch (lbf/in², psi)

Inf = 1.0E12  # a really large number
Null = 1/Inf  # a really small number

LunitTXT = "m"      # (Length) define basic-unit text for output
FunitTXT = "kN"     # (Force) define basic-unit text for output
RunitTXT = "rad"    # (Rotation) define basic-unit text for output
SunitTXT = "kPa"    # (Stress) define basic unit text for output
MunitTXT = "tonne"  # (Mass) define basic unit text for output
AunitTXT = "m/s²"   # (Acceleration) define basic unit text for output
VunitTXT = "m/s"    # (Velocity) define basic unit text for output
TunitTXT = "sec"    # (Time) define basic-unit text for output

print(
    f"""Basic OpenSees Output Units for this model:
    \t[{LunitTXT}]     for Length/Displacement,
    \t[{FunitTXT}]    for Force/Load,
    \t[{RunitTXT}]   for Angle/Rotation,
    \t[{SunitTXT}]   for Stress/Pressure,
    \t[{MunitTXT}] for Mass,
    \t[{AunitTXT}]  for Acceleration,
    \t[{VunitTXT}]   for Velocity,
    \t[{TunitTXT}]   for Time.\n""")


# ┌───────────────────────────────────────────────────────────────────────────────┐
# │                               Model Definition                                │
# └───────────────────────────────────────────────────────────────────────────────┘
# CORRECT combinations
# --------------------
# node numbering "simple" + with ANY constraints variant                ==> results in correct periods/frequencies (T1 = 0.5620 s)

# node numbering "problematicV1" + constraints "zeroLengthSection"      ==> results in correct periods/frequencies (T1 = 0.5620 s)
# node numbering "problematicV1" + constraints "zeroLength"             ==> results in correct periods/frequencies (T1 = 0.5620 s)
# node numbering "problematicV1" + constraints "twoNodeLink"            ==> results in correct periods/frequencies (T1 = 0.5620 s)

# node numbering "problematicV2" + constraints "zeroLengthSection"      ==> results in correct periods/frequencies (T1 = 0.5620 s)
# node numbering "problematicV2" + constraints "zeroLength"             ==> results in correct periods/frequencies (T1 = 0.5620 s)
# node numbering "problematicV2" + constraints "twoNodeLink"            ==> results in correct periods/frequencies (T1 = 0.5620 s)

# node numbering "problematicv3" + constraints "zeroLengthSection"      ==> results in correct periods/frequencies (T1 = 0.5620 s)
# node numbering "problematicV3" + constraints "zeroLength"             ==> results in correct periods/frequencies (T1 = 0.5620 s)
# node numbering "problematicV3" + constraints "twoNodeLink"            ==> results in correct periods/frequencies (T1 = 0.5620 s)


# INCORRECT combinations
# ----------------------
# node numbering "problematicV1" + constraints "equalDOF"               ==> results in INCORRECT periods/frequencies (T1 = 0.4189 s)
# node numbering "problematicV1" + constraints "rigidLink"              ==> results in INCORRECT periods/frequencies (T1 = 0.4189 s)

# node numbering "problematicV2" + constraints "equalDOF"               ==> results in INCORRECT periods/frequencies (T1 = 0.4956 s)
# node numbering "problematicV2" + constraints "rigidLink"              ==> results in INCORRECT periods/frequencies (T1 = 0.4956 s)

# node numbering "problematicV3" + constraints "equalDOF"               ==> results in INCORRECT periods/frequencies (T1 = 0.3245 s)
# node numbering "problematicV3" + constraints "rigidLink"              ==> results in INCORRECT periods/frequencies (T1 = 0.3245 s)


# Node numbering
# --------------
nodeNumbering = "problematicV1" # "simple", "problematicV1", "problematicV2", "problematicV3"

# Constraints type
# ----------------
constraints = "equalDOF" # "equalDOF", "rigidLink", "zeroLength", "twoNodeLink"

# Transient Excitation Type
# -------------------------
excitationType = "Uniform" # "Uniform", "MultipleSupport"
    
# ModelBuilder
# ------------
ops.wipe()
ops.model("basic", "-ndm", 1, "-ndf", 1)

# Node numering
# -------------
if nodeNumbering == "simple":
    """
    T1 = 0.5620 s
    
    With simple node numbering, everything is fine and validated with alternative software
    
    + Pairs of nodes -- node3+node4 after node2+node3 & node5+node6 after node2+node5 -- constrained
      with the equalDOF/rigidLink command one after the other in a sequence.
    """
    node1 = 1
    node2 = 2
    
    node3 = 3
    node4 = 4
         
    node5 = 5
    node6 = 6
    
elif nodeNumbering == "problematicV1":
    """
    T1 = 0.4189 sec
    
    This combination of node numbering gives equal frequencies and dynamic
    response as if the model only takes into account the mass in all nodes except the "Node6" !
    
    + Pairs of nodes -- node3+node4 after node2+node3 & node5+node6 after node2+node5 -- constrained
      with the equalDOF/rigidLink command one after the other in a sequence.
      
    + A consistent pattern was observed in this example: if the tag of the last constrained/secondary node in a sequence
      is smaller than that of its corresponding retained/primary node, the mass associated with the constrained/secondary
      node is excluded from the model.
    """
    node1 = 1
    node2 = 2
    
    node3 = 3
    node4 = 6
    
    node5 = 5
    node6 = 4
    # "Node6" with tag 4 is smaller than "Node5" with tag 5, therefore,
    # the mass at the "Node6" is not taken into account with the equalDOF/rigidLink command
    
elif nodeNumbering == "problematicV2":
    """
    T1 = 0.4956 sec
    
    This combination of node numbering gives equal frequencies and dynamic
    response as if the model only takes into account the mass in all nodes except the "Node4" !
        
    + Pairs of nodes -- node3+node4 after node2+node3 & node5+node6 after node2+node5 -- constrained
      with the equalDOF/rigidLink command one after the other in a sequence.
      
    + A consistent pattern was observed in this example: if the tag of the last constrained/secondary node in a sequence
      is smaller than that of its corresponding retained/primary node, the mass associated with the constrained/secondary
      node is excluded from the model.
    """
    node1 = 1
    node2 = 2
    
    node3 = 6
    node4 = 5
    # "Node4" with tag 5 is smaller than "Node3" with tag 6, therefore,
    # the mass at the "Node4" is not taken into account with the equalDOF/rigidLink command
    
    node5 = 3
    node6 = 4
    
elif nodeNumbering == "problematicV3":
    """
    T1 = 0.3245 sec
    
    This combination of node numbering gives equal frequencies and dynamic
    response as if the model doesn't take it into account the masses in both "Node4" and "Node6" !
        
    + Pairs of nodes -- node3+node4 after node2+node3 & node5+node6 after node2+node5 -- constrained
      with the equalDOF/rigidLink command one after the other in a sequence.
      
    + A consistent pattern was observed in this example: if the tag of the last constrained/secondary node in a sequence
      is smaller than that of its corresponding retained/primary node, the mass associated with the constrained/secondary
      node is excluded from the model.
    """
    node1 = 1
    node2 = 2
    
    node3 = 5
    node4 = 4
    # "Node4" with tag 4 is smaller than "Node3" with tag 5, therefore,
    # the mass at the "Node4" is not taken into account with the equalDOF/rigidLink command
    
    node5 = 6
    node6 = 3
    # "Node6" with tag 3 is smaller than "Node5" with tag 6, therefore
    # the mass at the "Node6" is not taken into account with the equalDOF/rigidLink command


ops.node(node1, 0.0, '-mass', 0.0) # Node 1
ops.node(node2, 0.0, '-mass', 2*tonne) # Node 2

ops.node(node3, 0.0, '-mass', 2*tonne) # Node 3
ops.node(node5, 0.0, '-mass', 2*tonne) # Node 5

ops.node(node4, 0.0, '-mass', 4*tonne) # Node 4 
ops.node(node6, 0.0, '-mass', 8*tonne) # Node 6 

# Duplicate nodes 3, 5 & 4, 6 without masses, for testing!
# --------------------------------------------------------
# ops.node(node4, 0.0, '-mass', 0.0) # Node 4 
# ops.node(node6, 0.0, '-mass', 0.0) # Node 6 

# Restraints
# ----------
ops.fix(node1, 1) # Node 1

# Materials & elements
# --------------------
ops.uniaxialMaterial('Elastic', 1, 2.25E3) # System stiffness
ops.uniaxialMaterial('Elastic', 999, 1E12) # "Rigid" material

ops.element('zeroLength', 1, *[node1, node2], '-mat', 1, '-dir', 1)


# Constraints
# -----------
if constraints == "equalDOF":
    ops.equalDOF(node2, node3, 1) # Nodes 2 & 3
    ops.equalDOF(node2, node5, 1) # Nodes 2 & 5
    
    ops.equalDOF(node3, node4, 1) # Nodes 3 & 4
    ops.equalDOF(node5, node6, 1) # Nodes 5 & 6
    
elif constraints == "rigidLink":
    ops.rigidLink("beam", node2, node3) # Nodes 2 & 3
    ops.rigidLink("beam", node2, node5) # Nodes 2 & 5
    
    ops.rigidLink("beam", node3, node4) # Nodes 3 & 4
    ops.rigidLink("beam", node5, node6) # Nodes 5 & 6
    
elif constraints == "zeroLength":
    ops.element('zeroLength', 121, *[node2, node3], '-mat', 999, '-dir', 1)
    ops.element('zeroLength', 122, *[node2, node5], '-mat', 999, '-dir', 1)
    
    ops.element('zeroLength', 123, *[node3, node4], '-mat', 999, '-dir', 1)
    ops.element('zeroLength', 124, *[node5, node6], '-mat', 999, '-dir', 1)
    
elif constraints == "twoNodeLink":
    ops.element('twoNodeLink', 121, *[node2, node3], '-mat', 999, '-dir', 1)
    ops.element('twoNodeLink', 122, *[node2, node5], '-mat', 999, '-dir', 1)
    
    ops.element('twoNodeLink', 123, *[node3, node4], '-mat', 999, '-dir', 1)
    ops.element('twoNodeLink', 124, *[node5, node6], '-mat', 999, '-dir', 1)

title("Model Build!")


# ┌───────────────────────────────────────────────────────────────────────────────┐
# │                               Modal Analysis                                  │
# └───────────────────────────────────────────────────────────────────────────────┘
ops.wipeAnalysis()
title("Running Modal Analysis!")

num_modes = 1
lam = ops.eigen("-fullGenLapack", num_modes)
# lam = ops.eigen(num_modes)

if len(lam) == 0:
    print("Try FullGenLapack\n")
    lam = ops.eigen("-fullGenLapack", num_modes)
elif len(lam) == 0:
    print("Try GenBandArpack\n")
    lam = ops.eigen("-genBandArpack", num_modes)
elif len(lam) == 0:
    print("Try SymmBandLapack\n")
    lam = ops.eigen("-symmBandLapack", num_modes)

# Extract the eigenvalues to the appropriate arrays
omega, freq, period = [], [], []

for index, item in enumerate(lam):
    omega.append(lam[index]**0.5)
    freq.append(lam[index]**0.5 / (2 * np.pi))
    period.append((2 * np.pi) / (lam[index]**0.5))
    
for i in range(0, num_modes):
    title(
        f"Mode #{i+1} ==> T{i+1} = {period[i]:.4f} sec, f{i+1} = {freq[i]:.4f} Hz, ω{i+1} = {omega[i]:.4f} rad/s")

ModalAnalysis = {"Omega": omega,
                 "Frequency": freq,
                 "Period": period}
    
# Check again with "modalProperties"
# ----------------------------------
ops.modalProperties("-print")


# ┌───────────────────────────────────────────────────────────────────────────────┐
# │                             Transient Analysis                                │
# └───────────────────────────────────────────────────────────────────────────────┘
# ops.wipeAnalysis()

ops.system("UmfPack") # UmfPack, FullGeneral
# ops.eigen("-fullGenLapack", num_modes)
dampRatio = 0.02
ops.modalDampingQ(dampRatio)

# The file with the ElCentro ground motion record is in the same Github repository as this OpenSees model 
EQfileName = "ElCentro"
GMdata = {"acc": np.loadtxt(EQfileName + ".txt"),
          "npts": 1560,
          "dt": 0.02*sec,
          }

ops.timeSeries('Path', 1, '-dt', GMdata["dt"], '-values', *GMdata["acc"], '-factor', g, "-prependZero")

if excitationType == "Uniform":
    ops.pattern('UniformExcitation', 1, 1, '-accel', 1, '-factor', 1)

elif excitationType == "MultipleSupport":
    ops.pattern('MultipleSupport', 1)
    gmTag = 101
    ops.groundMotion(gmTag, 'Plain', '-accel', 1, '-factor', 1)
    ops.imposedMotion(node1, 1, gmTag)


ctrlNodes = [node1, node2, node3, node4, node5, node6]
baseNodes = [node1]
title("Running Response History Analysis!")

ops.constraints("Transformation")
ops.numberer("RCM")
ops.test("NormDispIncr", 1.0E-8, 50)
ops.algorithm("Newton")
ops.integrator("Newmark", 0.5, 0.25)
ops.analysis("Transient")

tFinal = GMdata["npts"] * GMdata["dt"] + 5*sec  # integer of the number of steps needed
controlTime = ops.getTime()  # Start the controlTime

# Store some response
# -------------------
outputs = {
            "time": np.array([]),
            "gmAcc": np.array([]),
            "baseShear_x": np.array([]),
            }

for ctrlNs in ctrlNodes:
    outputs[f"disp_{ctrlNs}x"] = np.array([])
    outputs[f"vel_{ctrlNs}x"] = np.array([])
    outputs[f"acc_{ctrlNs}x"] = np.array([])
    
for Bnode in baseNodes:
    outputs[f"baseDisp_{Bnode}x"] = np.array([])

# Run the actual analysis now
while controlTime < tFinal: # Linear system!

    ok = ops.analyze(1, GMdata["dt"]) # Run a step of the analysis
    ops.reactions()

    for ctrlNs in ctrlNodes:
        outputs[f"disp_{ctrlNs}x"] = np.append(
            outputs[f"disp_{ctrlNs}x"], ops.nodeDisp(ctrlNs, 1))
        outputs[f"vel_{ctrlNs}x"] = np.append(
            outputs[f"vel_{ctrlNs}x"], ops.nodeAccel(ctrlNs, 1))
        outputs[f"acc_{ctrlNs}x"] = np.append(
            outputs[f"acc_{ctrlNs}x"], ops.nodeVel(ctrlNs, 1))

    outputs["time"] = np.append(outputs["time"], controlTime)
    outputs["gmAcc"] = np.append(outputs["gmAcc"], ops.getLoadFactor(1))

    nodeList_baseShear_x = []
    
    for node in baseNodes:
        outputs[f"baseDisp_{node}x"] = np.append(
            outputs[f"baseDisp_{node}x"], ops.nodeDisp(node, 1))
        nodeList_baseShear_x.append(f"-ops.nodeReaction({node}, 1) ")
        
    nodeList_baseShear_x = "".join(nodeList_baseShear_x)

    outputs["baseShear_x"] = np.append(outputs["baseShear_x"], eval(nodeList_baseShear_x))

    controlTime = ops.getTime()  # Update the control time
    
print(f"Final State: {controlTime:.3f} at {controlTime:.3f} of {tFinal:.3f} seconds")

# Plot the response
# -----------------
plot_fig(figure_size=[15, 8], font_size=11, use_latex=False)

lineColors = [
    (0.0,    0.4470, 0.7410), # blue
    (0.8500, 0.3250, 0.0980), # red-orange
    (0.9290, 0.6940, 0.1250), # yellow
    (0.4940, 0.1840, 0.5560), # purple
    (0.4660, 0.6740, 0.1880), # green
    (0.3010, 0.7450, 0.9330)  # light blue
]

if excitationType == "Uniform":
    for i in range(1, len(ctrlNodes)):
        node_index = ctrlNodes[i]
        plt.plot(outputs["time"], outputs[f"disp_{node_index}x"]/mm, color=lineColors[i], linewidth=3-i*0.5, label=f"Node {ctrlNodes[i]} in x-dir")

elif excitationType == "MultipleSupport":
    for i in range(0, len(ctrlNodes)):
        node_index = ctrlNodes[i]
        plt.plot(outputs["time"], outputs[f"disp_{node_index}x"]/mm, color=lineColors[i], linewidth=3-i*0.5, label=f"Node {ctrlNodes[i]} in x-dir")

plt.xlabel("Time [s]")
plt.ylabel("Displacement [mm]")
figTitle = r"Node numbering: $\mathbf{" + nodeNumbering + "}$, Constraints: $\mathbf{" + constraints + "}$"
plt.title(figTitle)
plt.legend(loc="lower right")
plt.show()

# Wipe model!
ops.wipe()
