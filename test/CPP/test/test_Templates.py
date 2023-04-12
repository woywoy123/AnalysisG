import math

def test_particle_template():

    from AnalysisG.Templates import ParticleTemplate 
    
    def Px(pt, phi): return pt*math.cos(phi)
    def Py(pt, phi): return pt*math.sin(phi)
    def Pz(pt, eta): return pt*math.sinh(eta)

    class Particle(ParticleTemplate):
        def __init__(self):
            ParticleTemplate.__init__(self)
        
    x = ParticleTemplate()
    y = ParticleTemplate()
    
    assert x.hash == y.hash
    assert len(x.hash) == 18
    
    # Testing setter function
    vals = [207050.75, 0.5622375011444092, 2.262759208679199, 296197.3125]
    val = [Px(vals[0], vals[2]), Py(vals[0], vals[2]), Pz(vals[0], vals[1]), vals[3]]
    
    t = Particle()
    t.pt = vals[0]
    t.eta = vals[1]
    t.phi = vals[2]
    t.e = vals[3]
    
    # Assert Setter and Getter Functions 
    assert t.pt == vals[0]
    assert t.eta == vals[1]
    assert t.phi == vals[2]
    assert t.e == vals[3]
   
    # Assert Getter Function with transform to cartesian 
    assert t.px == val[0]
    assert t.py == val[1]  
    assert t.pz == val[2]
    
    # Assert the reverse transformation to polar 
    t_ = Particle()
    t_.px = val[0]
    t_.py = val[1]
    t_.pz = val[2]
    t_.e = val[3]

    # Assert the Setter of Cartesian
    assert t_.px == val[0]
    assert t_.py == val[1]
    assert t_.pz == val[2]
    assert t_.e == val[3]

    assert t_.pt == vals[0]
    assert t_.eta == vals[1]
    assert t_.phi == vals[2]
    assert t_.e == vals[3]
   
    # Assert Arithmitic 
    t += t_
    assert t.px == val[0]*2
    assert t.py == val[1]*2
    assert t.pz == val[2]*2
    assert t.e == val[3]*2
   
    # Bulk Summation
    x = [t_, t_, t_, t_, t_]
    l = len(x)
    k = sum(x)
    assert k.px == t_.px*l
    assert k.py == t_.py*l
    assert k.pz == t_.pz*l
    assert k.e  == t_.e*l

    assert k.px == t.px*l/2
    assert k.py == t.py*l/2
    assert k.pz == t.pz*l/2
    assert k.e  == t.e*l/2
    
    # Check hashing 
    assert k.hash != t.hash
    assert t_.hash != t.hash
    
    # Check if equal 
    assert k == k
    assert k != t_
    assert t_ != t
    
    x = [t, t_, t, t_]
    assert len(set(x)) == 2
    assert x.index(t_) == 1
    
    # Test for memory leak
    for i in range(10000000):
        t_ = Particle()
        t_.px = val[0]
        t_.py = val[1]
        t_.pz = val[2]
        t_.e = val[3]

if __name__ == "__main__":
    test_particle_template()
