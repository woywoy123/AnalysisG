Double Leptonic Resonance Decay Analysis
****************************************
This small analysis script aims to reconstruct the mass of the heavy vector/scalar H/Z boson, using a primitive grouping algorithm. 

Selection Criteria
__________________
Events are required to have at least two leptons and four b-jets. 

Grouping Strategy
_________________
To seed the grouping, the two leading order b-tagged partons are used to indicate the presence of at least 2-tops. 
These are considered as the `Resonance` top pairs, which are emitted from the heavy resonance. 
If any of these seeded b-tagged partons are closest to an adjacent quark parton (or jet), then the associated object will be marked as a hadronically decaying top quark. 
Whereas, if the $\Delta$R is lowest between a leptonic object, then the b-tagged object will be tagged as originating from a leptonically decaying top quark. 
This procedure is repeated for all b-tagged objects, until no further clustering is possible. 
If there are two leptonically decaying tops, then the double neutrino reconstruction algorithm will be invoked to restore the neutrino momenta. 


.. ## Single Leptonic (Bruce's Suggestion):
   ### Selection criteria:
    - Assume that the leptonically decaying top is the spectator.
    - Assume that the two hardest b-tagged jets (PT) are the Z' resonant tops.
    - Match two additional jets to each using $\Delta$R.
    - Throw away any remaining jets, these are considered to belong to the second spectator top.


