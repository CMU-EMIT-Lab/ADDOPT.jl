
function traj_to_lines(id::GMAWDynamicsPrescribed, Δt, U)
    Nk = length(Δt)
    @assert Nk == length(U)
    @assert NK == length(id.p̄)

    t = cumsum(Δt)
    p̄ = id.p̄
    p̄ = vcat(p̄, [p̄[end]])

    dr = [norm(p̄[k+1] - p̄[k]) for k in 1:Nk]
    TS = dr ./ Δt
    WFS = [u[1] for u in U]

    lines = []
    push!(lines, [p̄[1]; TS[1]; WFS[1]])

    for (p, ts, wfs) in zip(p̄, TS, WFS)
        if [ts; wfs] != lines[end][4:5]
            push!(lines, [p; ts; wfs])
        end
    end

    return lines
end

function lines_to_rapid(lines)
    s = ""

    was_on = false
    for line in lines
        x, y, z, ts, wfs = line

        if !was_on && wfs > 0
            s *= "WArcLStart [[$(round(1e3x; digits=2)), $(round(1e3y; digits=2)), $(round(1e3z; digits=2))], [0.000260569,-0.674916,-0.737894,0.00491837], [-1,-1,-3,0], [9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]],v100\\V:=$(round(1e3ts; digits=2)),sePrint,bdPrint,fine,tPrintGun\\WObj:=obParamWall\n"
            was_on = true
        end

        if was_on && wfs == 0
            s *= "WArcLEnd [[$(round(1e3x; digits=2)), $(round(1e3y; digits=2)), $(round(1e3z; digits=2))], [0.000260569,-0.674916,-0.737894,0.00491837], [-1,-1,-3,0], [9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]],v100\\V:=$(round(1e3ts; digits=2)),fine,tPrintGun\\WObj:=obParamWall;\n"
            was_on = false
        end

        if wfs > 0
            s *= "WArcL [[$(round(1e3x; digits=2)), $(round(1e3y; digits=2)), $(round(1e3z; digits=2))], [0.000260569,-0.674916,-0.737894,0.00491837], [-1,-1,-3,0], [9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]],v100\\V:=$(round(1e3ts; digits=2)),z0,tPrintGun\\WObj:=obParamWall;\n"
        else
            s *= "MoveL [[$(round(1e3x; digits=2)), $(round(1e3y; digits=2)), $(round(1e3z; digits=2))], [0.000260569,-0.674916,-0.737894,0.00491837], [-1,-1,-3,0], [9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]],v100\\V:=$(round(1e3ts; digits=2)),fine,tPrintGun\\WObj:=obParamWall;\n"
        end
    end

end