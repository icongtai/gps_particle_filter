PFFG, an improved particle filter algorithm for GPS data, including speed data and poisition data(latitude and longitude). 
PFFG is good at filting singular value, but will add some random fluctuation to normal data.This defection almost has no effect on speed data, it has no effect on calculating acceleration value.But it may affect the accuracy on track, for drawing track on maps demands high quality posisiton data and has little tolerance even on small random error.
Two functions in the package of PFFG, 
To use PFFG, the only thing you need to do is importing the package, pf_speed function can filter speed data and pf_poision function can filter poision data(latitude or longitude).
For more details, you can see example in 'function_test.py'