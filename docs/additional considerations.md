### Part 1: Additional Algorithmic Nuances

- **Initialization Vector ($x^0$ Specification):**
    
    - _The Detail:_ In MLEM, the initial guess vector $x^0$ should not be a flat or random array. Shevelev (2013) specifies initializing with the normalized measured data:
        $$x^0 = \frac{y}{\Vert{}H y\Vert{}}$$
        
    - _Impact:_ Prevents MLEM from starting in an unphysical state and speeds up convergence.
        
- **Monte Carlo Uncertainty Propagation:**
      
    - _The Detail:_ A single MLEM run produces a point estimate without statistical error bars.
        
    - _The Method:_ To compute error bars on $x$:
          
        1. Generate $100+$ synthetic copies of $y$ by drawing random Poisson samples: $y^{(k)} \sim \text{Poisson}(y)$.
            
        2. Run the MLEM deconvolution on each $y^{(k)}$ to yield $x^{(k)}$.
            
        3. Compute the mean and standard deviation across all $x^{(k)}$ vectors for every energy bin.
            
- **Background Subtraction vs. Forward Modeling:**
    
      
    - _The Detail:_ Hard X-ray measurements contain neutron-induced background, crystal activation, and cosmic background.
        
          
        
    - _The Method:_ Either subtract the background spectrum $b$ prior to deconvolution ($y_{\text{clean}} = y - b$), or incorporate $b$ directly into the forward step ($y = H x + b$).
        

### Part 2: Additional Experimental & Physical Nuances

- **Relativistic Bremsstrahlung Angular Anisotropy:**
    
    - _The Detail:_ Relativistic runaway electrons ($\text{MeV}$ energies) emit Bremsstrahlung in a tight forward cone of half-angle $\theta \approx 1/\gamma$.
        
    - _Impact:_ $H_e$ depends heavily on the electron pitch angle relative to the detector sightline (e.g., ADITYA-U's tangential limiter view vs. vertical views). $H_e$ is not just an energy lookup table; it embeds the viewing geometry.
        
- **Pulse Pile-up at High Count Rates:**
    
    - _The Detail:_ At high HXR fluxes (exceeding $2.5 \text{ Lakhs counts/sec}$ or $250\text{ kCPS}$), two low-energy photons striking the crystal within the decay time ($16\text{ ns}$ for $\text{LaBr}_3$) register as a single high-energy event.
        
    - _Impact:_ Pile-up creates an artificial high-energy tail. The raw spectrum must be pile-up corrected or collimated before running MLEM.
        
- **Absolute Runaway Current ($I_{\text{RE}}$) Scaling:**
    
    - _The Detail:_ Shape deconvolution provides relative energy distribution shapes $f(E)$. To convert $f(E)$ into total runaway electron current $I_{\text{RE}}$, integrate the electron flux across the plasma volume $V_{\text{obs}}$ seen by the detector chord:
        
        $$I_{\text{RE}} \propto e \int_{E_{\text{thresh}}}^{E_{\text{max}}} v(E) f(E) dE$$
        
    - _Requirement:_ Requires absolute detector efficiency and geometric solid angle ($\Omega / 4\pi$), rather than unit-normalized columns.