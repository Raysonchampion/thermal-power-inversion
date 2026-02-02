https://1drv.ms/f/c/13ab85533c58e493/IgDSMrvx17vPRKESUcubxwiOAfy88ii8fJq7fQRN5KMisUU  
1.
2 folders here:
<input>
  input data of pbthermal 
  [File]:
  initial_temp_map files(30) X core_power_trace files(100)
  [Naming rule]: 
  temp_map.${initial temperature map id}.dat
  core.steady${core power trace id}.ptrace.yaml

<output>
  output result of pbthermal
  [File]:
  result_summary files(3000) 
  [Naming rule]: 
  result_summary${initial temperature map id}-${core power trace id}.dat

  In 1 result_summary*-*.dat, 
  temperature results of 54 power blocks(surface sensor contains comlpete power block)
  temperature results of 54 point senors(center point of power block)

2.
2 folders here:
<input>
  input data of pbthermal 
  [File]:
  initial_temp_map files(30) X core_power_trace files(20) X boundary condition map(5)
  [Naming rule]: 
  temp_map.${initial temperature map id}.dat
  core.trans${core power trace id}.ptrace.yaml
  boundary200/500/700/1000/1300.bdmap

<output>
  output result of pbthermal
  [File]:
  result_summary files(3000) 
  [Naming rule]: 
  result_summary${initial temperature map id}-${core power trace id}-HTC${HTC value}.dat

  In 1 result_summary.dat, 
  temperature results of 54 power blocks(surface sensor contains comlpete power block)
  temperature results of 54 point senors(center point of power block)
3.
2 folders here:
<T>
 [File]:
 workload conditions(19)x times(100)x
 [Naming rule]:
 CPU_i7_${workload condition}_thermal_maps_{id}.csv
<P>
 [File]:
 workload conditions(19)x times(100)x
 [Naming rule]:
 CPU_i7_${workload condition}_power_maps_{id}.csv


