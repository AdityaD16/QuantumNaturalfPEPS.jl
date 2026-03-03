# This file defines custom operators for ITensors.jl, specifically the ZtoY and YtoZ transformations. These operators are used in the PEPS simulations to transform between different bases.

const ZtoY = (1/sqrt(2)) * [1 1; 1 -1] * [1 0; 0 -1im]
const YtoZ = [1 0; 0 1im] * (1/sqrt(2)) * [1 1; 1 -1]

ITensors.op(::ITensors.SiteTypes.OpName"ZtoY", ::ITensors.SiteTypes.SiteType"Qubit") = ZtoY
ITensors.op(::ITensors.SiteTypes.OpName"YtoZ", ::ITensors.SiteTypes.SiteType"Qubit") = YtoZ
ITensors.op(::ITensors.SiteTypes.OpName"ZtoY", ::ITensors.SiteTypes.SiteType"S=1/2") = ZtoY
ITensors.op(::ITensors.SiteTypes.OpName"YtoZ", ::ITensors.SiteTypes.SiteType"S=1/2") = YtoZ