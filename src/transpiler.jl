using LinearAlgebra

function traj_to_lines(id::GMAWDynamicsFullyPrescribed, t, TS, WFS, Trim)
    Nk = length(t)
    @assert Nk == length(TS)
    @assert Nk == length(id.p̄)

    p̄ = id.p̄
    p̄ = vcat(p̄, [p̄[end]])

    lines = []
    push!(lines, [p̄[1]; TS[1]; WFS[1]; Trim[1]; 0.0])

    for (time, p, ts, wfs, trim) in zip(t, p̄, TS, WFS, Trim)
        if ((trim > 0) || (trim == 0 && lines[end][6] > 0)) && norm([ts; wfs; trim] .- lines[end][4:6], Inf) >= 1e-4
            push!(lines, [p; ts; wfs; trim; time])
        end
    end

    for k in lastindex(lines):-1:2
        lines[k][end] -= lines[k-1][end]
    end

    lines[1][end] = 3.0

    return lines
end

function lines_to_rapid(lines)
    s = ""
    weave_length = 2
    weave_width = 0.1
    dwell_left = 0
    dwell_right = 0
    weld_mode = 329
    track_ref_current = 135

    was_on = false
    for line in lines
        x, y, z, ts, wfs, trim, time = line
        x = round(1e3 * x, digits=3)
        y = round(1e3 * y, digits=3)
        z = round(1e3 * z, digits=3)
        ts = round(1e3 * ts, digits=3)
        wfs = round(1e3 * wfs, digits=3)
        trim = round(trim, digits=3)
        time = round(time, digits=4)


        if !was_on && wfs > 0
            s *= "\n"
            s *= (" "^8) * "exp_bead_data:=[$ts,[$weave_length,$weave_width,$dwell_left,$dwell_right],[$weld_mode,0,$trim,$wfs,0,0,0,0,0],$track_ref_current];\n"
            s *= (" "^8) * "dest.trans:=[$x,$y,$z];\n"
            s *= (" "^8) * "MoveL dest,v100\\T:=$(time),fine,tPrintGun\\WObj:=obParamWall;\n"
            s *= (" "^8) * "WArcLStart dest,v100,exp_start_end,exp_bead_data,fine,tPrintGun\\WObj:=obParamWall;\n"
            was_on = true
        elseif was_on && wfs == 0
            s *= (" "^8) * "dest.trans:=[$x,$y,$z];\n"
            s *= (" "^8) * "WArcLEnd dest,v100,\\Bead:=exp_bead_data,fine,tPrintGun\\WObj:=obParamWall;\n\n"
            was_on = false
        elseif wfs > 0
            s *= (" "^8) * "exp_bead_data:=[$ts,[$weave_length,$weave_width,$dwell_left,$dwell_right],[$weld_mode,0,$trim,$wfs,0,0,0,0,0],$track_ref_current];\n"
            s *= (" "^8) * "dest.trans:=[$x,$y,$z];\n"
            s *= (" "^8) * "WArcL dest,v100,\\Bead:=exp_bead_data,z0,tPrintGun\\WObj:=obParamWall;\n"
        end
            # else
        #     s *= (" "^8) * "dest.trans:=[$x,$y,$z];\n"
        #     s *= (" "^8) * "MoveL dest,v100\\T:=$(time),fine,tPrintGun\\WObj:=obParamWall;\n"
        # end
    end

    x, y, z, ts, wfs, trim, time = lines[end]
    x = round(1e3 * x, digits=3)
    y = round(1e3 * y, digits=3)
    z = round(1e3 * z, digits=3)
    s *= "\n"
    s *= (" "^8) * "dest.trans:=[$x,$y,$(z+200)];\n"
    s *= (" "^8) * "MoveL dest,v100,fine,tPrintGun\\WObj:=obParamWall;\n"

    return s
end