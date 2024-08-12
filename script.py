'''
PPGEE - Departamento de Engenharia Eletrica e de Computacao - UFBA
UR5: Cinematica Direta, Inversa e Planejamento de Trajetoria
Alunos: Andre Paiva Conrado Rodrigues e Gabriel Lucas Nascimento Silva
Disciplina: Sistemas Roboticos - 2024.1
Data: 11 ago 2024
'''

#-----------------------------------------------#
#----------Importacao de Dependencias-----------#
#-----------------------------------------------#

import numpy as np
from time import sleep

#-----------------------------------------------#
#---------------Variaveis Globais---------------#
#-----------------------------------------------#
SIGNAL = True

FLAG_DK = True # Flag que indica se a validacao da C.D. foi concluida
COUNTER_DK = 0 # Contador de casos de teste da C.D. que passaram por validacao

# Array de casos de teste da C.D (J1, J2, J3, J4, J5, J6)
TESTES_DH = np.array([
                        [   0,      0,      0,      0,      0,       0    ],
                        [   30,     60,     90,     0,      0,       0    ],
                        [   0,      0,      0,      30,     60,      90   ],
                        [   -45,    -30,    60,     15,     -30,     75   ],
                        [   45,     -90,    75,     30,     -25,     -90  ],
                        [   39,     42,     -80,    9,      -112,    13   ],
                        [   -158,   97,     45,    -138,    125,     -147 ],
                        [   130,    -20,    -50,    -45,    25,      110  ],
                    ])

NUM_TESTES_DH = TESTES_DH.shape[0] # Quantidade de casos de teste da C.D.

#-----------------------------------------------#
#-------------Interacao CoppeliaSim-------------#
#-----------------------------------------------#

# Inicializacao
def sysCall_init():
    sim = require('sim')
    joints = get_joints()
    for joint in joints:
        sim.setJointMode(joint, sim.jointmode_kinematic, 1)

# Validacoes implementadas no sensing
def sysCall_sensing():
    global FLAG_DK, COUNTER_DK, TESTES_DH, NUM_TESTES_DH

    if(not FLAG_DK):
        dk_validate(TESTES_DH, COUNTER_DK)
        COUNTER_DK += 1
        if(COUNTER_DK >= NUM_TESTES_DH):
            FLAG_DK = True
    
    print(ik_get_all_theta())
    print(read_joints_sensors())
    print()

# Atuacao step-by-step
def sysCall_actuation():
    global SIGNAL
    
    if(FLAG_DK):
        joints = get_joints()
        for joint in joints:
            orient = sim.getJointPosition(joint)
            if(orient > np.pi/2):
                SIGNAL = False
            elif(orient < -np.pi/2):
                SIGNAL = True
            if(SIGNAL):
                orient += 0.005
            else:
                orient -= 0.005
            sim.setJointPosition(joint, orient)

#-----------------------------------------------#
#--------------Funcoes Auxiliares---------------#
#-----------------------------------------------#

# Captura dos handlers das juntas
def get_joints():
    joints = []
    for i in range(6):
        joint = '/UR5/joint'
        if(i > 0):
            joint = '/UR5' + i * '/link/joint'
        joints.append(sim.getObject(joint))
    return joints

# Leitura de sensores das juntas
def read_joints_sensors():
    joints = get_joints()
    theta = []
    for joint in joints:
        theta.append(sim.getJointPosition(joint))
    return theta

# Limita angulo entre -pi e pi
def wrap_angle(angle):
    return np.atan2(np.sin(angle),np.cos(angle))

#-----------------------------------------------#
#---------Funcoes de Cinematica Direta----------#
#-----------------------------------------------#

# Tabela DH (a, alpha, d, theta)
def dk_get_dh():
    theta = read_joints_sensors()
    dh = np.array([ [    0,             -np.pi/2,   89.16e-3,  theta[0] + np.pi/2  ], # A1
                    [    425e-3,        0,          0,         theta[1] - np.pi/2  ], # A2
                    [    392.25e-3,     0,          109.15e-3, theta[2]            ], # A3
                    [    0,             -np.pi/2,   0,         theta[3] - np.pi/2  ], # A4
                    [    0,             np.pi/2,    94.65e-3,  theta[4]            ], # A5
                    [    0,             0,          82.3e-3,   theta[5]            ], # A6
                 ])
    return dh
    
# Funcao auxiliar para montar matriz Ai
def dk_mount_ai_matrix(dh_line):
    ai = dh_line[0]
    di = dh_line[2]
    s_alp = np.sin(dh_line[1])
    c_alp = np.cos(dh_line[1])
    s_the = np.sin(dh_line[3])
    c_the = np.cos(dh_line[3])
    matrix = np.array([ [ c_the,    -s_the*c_alp,   s_the*s_alp,    ai*c_the ],
                        [ s_the,    c_the*c_alp,    -c_the*s_alp,   ai*s_the ],
                        [ 0,        s_alp,          c_alp,          di       ],
                        [ 0,        0,              0,              1        ]])
    return matrix

# Matriz Ai
def dk_get_ai(i):
    if i < 1 or i > 6:
        raise Exception('i deve estar entre 1 e 6')
    else:
        dh_line = dk_get_dh()[i-1]
        matrix = dk_mount_ai_matrix(dh_line)
        return matrix
        
# Matriz de transformacao completa
def dk_get_transformation_matrix():
    t_matrix = np.identity(4)
    for i in range(1, 7):
        t_matrix = np.matmul(t_matrix, dk_get_ai(i))
    return t_matrix

# Posicao da garra
def dk_get_end_effector_pose():
    base_matrix = sim.getObjectMatrix(sim.getObject("/UR5"))
    base_matrix = np.array([base_matrix[0:4],
                            base_matrix[4:8],
                            base_matrix[8:12],
                            [0, 0, 0, 1]])
    t_matrix = dk_get_transformation_matrix()
    end_effector_matrix = np.matmul(base_matrix, t_matrix)
    return end_effector_matrix

# Validacao Cinematica Direta
def dk_validate(test_cases, num_teste):
    TOLERANCE = 0.03    # Error tolerance in meters

    end_effector = sim.getObject("/UR5/connection")
    joints = get_joints()
    fail_count = 0

    print(f"========================================================")
    print(f"Validacao - Cinematica Direta - Caso teste {num_teste+1}")
    print(f"========================================================")

    for joint, value in zip(joints, test_cases[num_teste]):
        sim.setJointPosition(joint, value*np.pi/180.0)
    end_pose = dk_get_end_effector_pose()
    end_ground = sim.getObjectPosition(end_effector)
    end_diff = [end_pose[0][3] - end_ground[0],
                end_pose[1][3] - end_ground[1], 
                end_pose[2][3] - end_ground[2], ]
    print(f"Angulos das Juntas: {test_cases[num_teste]}")
    print(f"Calculo: X={end_pose[0][3]:.7f} \t Y={end_pose[1][3]:.7f} \t Z={end_pose[2][3]:.7f}")
    print(f"Truth:   X={end_ground[0]:.7f} \t Y={end_ground[1]:.7f} \t Z={end_ground[2]:.7f}")
    print(f"Diff:    X={end_diff[0]:.7f} \t Y={end_diff[1]:.7f} \t Z={end_diff[2]:.7f}")
    for i in range(len(end_diff)):
        if abs(end_diff[i]) > TOLERANCE:
            fail_count += 1
            axis = ''
            if(i == 0):
                axis = 'X'
            elif(i == 1):
                axis = 'Y'
            elif(i == 2):
                axis = 'Z'
            print(f"FALHA: Diferenca de posicao do eixo {axis} fora da faixa de tolerancia")
    print(f"--------------------------------------------------------")
    
    if(fail_count == 0):
        print(f"Caso teste {num_teste+1} validado com exito.")
    else:
        print(f"Caso teste {num_teste+1} FALHOU com {fail_count} erros.")
    print(f"========================================================")
    print()

    sleep(1)
    
    if(num_teste >= test_cases.shape[0] - 1):
        for joint, value in zip(joints, test_cases[num_teste]):
            sim.setJointPosition(joint, 0)
    else:
        for joint, value in zip(joints, test_cases[num_teste+1]):
            sim.setJointPosition(joint, value*np.pi/180.0)

#-----------------------------------------------#
#--------Funcoes de Cinematica Inversa----------#
#-----------------------------------------------#

def ik_get_all_theta():
    return np.array ([
                        ik_get_theta1(), ik_get_theta2(), ik_get_theta3(),
                        ik_get_theta4(), ik_get_theta5(), ik_get_theta6(),
                    ])

# Theta 1 OK (?)
def ik_get_theta1(shoulder_left = True):
    dh = dk_get_dh()
    wrist = np.identity(4)
    for i in range(1, 6):
        wrist = np.matmul(wrist, dk_get_ai(i))
    p5x = wrist[0][3]
    p5y = wrist[1][3]
    d3 = dh[2][2]
    if(shoulder_left):
        theta1 = np.atan2(p5y, p5x) + np.acos(d3/np.sqrt(p5x**2 + p5y**2)) + np.pi
    else:
        theta1 = np.atan2(p5y, p5x) - np.acos(d3/np.sqrt(p5x**2 + p5y**2)) + np.pi
    return wrap_angle(theta1)

def ik_get_theta2():
    dh = dk_get_dh()
    elbow = np.identity(4)
    for i in range(2, 5):
        elbow = np.matmul(elbow, dk_get_ai(i))
    p4x = elbow[0][3]
    p4z = elbow[2][3]
    p4xz = np.sqrt(p4x**2 + p4z**2)
    a3 = dh[2][0]
    theta3 = ik_get_theta3()
    theta2 = np.atan2(-p4z, -p4x) - np.asin((-a3*np.sin(theta3))/p4xz) + np.pi/2
    return wrap_angle(theta2)

def ik_get_theta3(elbow_up = True):
    dh = dk_get_dh()
    elbow = np.identity(4)
    for i in range(2, 5):
        elbow = np.matmul(elbow, dk_get_ai(i))
    p4xz = np.sqrt((elbow[0][3]**2) + (elbow[2][3]**2))
    a2 = dh[1][0]
    a3 = dh[2][0]
    theta3 = wrap_angle(np.acos((p4xz**2 - a2**2 - a3**2)/(2*a2*a3)))
    if(not elbow_up):
        theta3 *= -1
    return theta3

# Theta 4 OK
def ik_get_theta4():
    t34 = dk_get_ai(4)
    X4y = t34[1][0]
    X4x = t34[0][0]
    theta4 = wrap_angle(np.atan2(X4y, X4x) + np.pi/2)
    return theta4

# Theta 5 OK (?)
def ik_get_theta5(wrist_up = True):
    dh = dk_get_dh()
    t_matrix = dk_get_transformation_matrix()
    p6x = t_matrix[0][3]
    p6y = t_matrix[1][3]
    d3 = dh[2][2]
    d6 = dh[5][2]
    theta1 = ik_get_theta1() - np.pi/2
    theta5 = wrap_angle(np.acos(((p6x*np.sin(theta1))-(p6y*np.cos(theta1))-d3)/d6))
    if(not wrist_up):
        theta5 *= -1
    return theta5

def ik_get_theta6():
    t_inv_matrix = np.identity(4)
    for i in range(6, 0, -1):
        t_inv_matrix = np.matmul(t_inv_matrix, dk_get_ai(i))
    X0x = t_inv_matrix[0][0]
    X0y = t_inv_matrix[1][0]
    Y0x = t_inv_matrix[0][1]
    Y0y = t_inv_matrix[1][1]
    theta1 = ik_get_theta1() - np.pi/2
    theta5 = ik_get_theta5()
    atan_first = (-X0y*np.sin(theta1) + Y0y*np.cos(theta1))/np.sin(theta5)
    atan_second = (X0x*np.sin(theta1) - Y0x*np.cos(theta1))/np.sin(theta5)
    theta6 = np.atan2(atan_first, atan_second)
    return wrap_angle(theta6)
