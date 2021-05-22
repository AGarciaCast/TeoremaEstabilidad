-- Welcome to Macaulay2 !
-- In this window you may type in Macaulay2 commands
-- and have them evaluated by the server.

-- Evaluate a line or selection by typing Shift+Enter
-- or by clicking on Evaluate.

-- What follows are some examples.
-- you can erase it all with e.g. Ctrl-A + Delete

needsPackage "SimplicialComplex"
needsPackage "Visualize"

-- Definir vertices
R = ZZ[a..g]

-- Definir complejo
D = simplicialComplex {a*b*c,a*b*d,a*e*f,a*g}

-- Visualizar
openPort "8080"
H = visualize D 
