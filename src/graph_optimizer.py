import numpy as np
from src.truss_fem import solve_truss

def structural_mass(nodes, elements, A, rho):
    lengths = np.linalg.norm(nodes[elements[:, 1]] - nodes[elements[:, 0]], axis=1)
    return rho*A*np.sum(lengths)

def greedy_optimize(nodes,elements,E,A,rho,loads,fixed_dofs,target_mass):
    current=elements.copy(); history=[]
    while True:
        result=solve_truss(nodes,current,E,A,loads,fixed_dofs)
        mass=structural_mass(nodes,current,A,rho)
        compliance=result['compliance']
        history.append({'mass':mass,'compliance':compliance,'elements':current.copy()})
        if mass<=target_mass: break
        best_score=np.inf;best_elements=None
        for k in range(len(current)):
            candidate=np.delete(current,k,axis=0)
            try: cr=solve_truss(nodes,candidate,E,A,loads,fixed_dofs)
            except ValueError: continue
            cm=structural_mass(nodes,candidate,A,rho)
            score=(cr['compliance']-compliance)/(mass-cm)
            if score<best_score: best_score=score;best_elements=candidate
        if best_elements is None:
            print('No further stable member removals.');break
        current=best_elements
    return current,history

def compliance_constrained_optimize(nodes,elements,E,A,rho,loads,fixed_dofs,max_compliance_increase=0.05):
    current=elements.copy();history=[]
    limit=solve_truss(nodes,current,E,A,loads,fixed_dofs)['compliance']*(1+max_compliance_increase)
    while True:
        result=solve_truss(nodes,current,E,A,loads,fixed_dofs)
        mass=structural_mass(nodes,current,A,rho); c=result['compliance']
        history.append({'mass':mass,'compliance':c,'elements':current.copy()})
        best_score=np.inf; best=None
        for k in range(len(current)):
            candidate=np.delete(current,k,axis=0)
            try: cr=solve_truss(nodes,candidate,E,A,loads,fixed_dofs)
            except ValueError: continue
            if cr['compliance']>limit:continue
            cm=structural_mass(nodes,candidate,A,rho)
            score=(cr['compliance']-c)/(mass-cm)
            if score<best_score: best_score=score;best=candidate
        if best is None:break
        current=best
    return current,history

def force_based_optimize(nodes,elements,E,A,rho,loads,fixed_dofs,max_compliance_increase=0.05):
    current=elements.copy();history=[]
    limit=solve_truss(nodes,current,E,A,loads,fixed_dofs)['compliance']*(1+max_compliance_increase)
    while True:
        result=solve_truss(nodes,current,E,A,loads,fixed_dofs)
        mass=structural_mass(nodes,current,A,rho)
        history.append({'mass':mass,'compliance':result['compliance'],'elements':current.copy()})
        best=None
        for k in np.argsort(np.abs(result['axial_forces'])):
            candidate=np.delete(current,k,axis=0)
            try: cr=solve_truss(nodes,candidate,E,A,loads,fixed_dofs)
            except ValueError:continue
            if cr['compliance']<=limit:
                best=candidate;break
        if best is None:break
        current=best
    return current,history
