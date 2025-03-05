
export LinearConstraint
struct LinearConstraint{T<:AbstractFloat,V<:AbstractVector{T},M<:AbstractMatrix{T}} <: Constraint{T}
    A_x_eq::M
    b_x_eq::V
    A_u_eq::M
    b_u_eq::V
    A_x_ineq::M
    b_x_ineq::V
    A_u_ineq::M
    b_u_ineq::V
end

nc_eq(constraint::LinearConstraint)::Int = length(constraint.b_x_eq) + length(constraint.b_u_eq)
nc_ineq(constraint::LinearConstraint)::Int = length(constraint.b_x_ineq) + length(constraint.b_u_ineq)

function eq_constraint!(constraint::LinearConstraint, c, x, u)
    A_x, A_u = constraint.A_x_eq, constraint.A_u_eq
    b_x, b_u = constraint.b_x_eq, constraint.b_u_eq
    ncx = length(b_x)
    ncu = length(b_u)

    c_x = @view c[1:ncx]
    c_u = @view c[(ncx+1):(ncx+ncu)]

    mul!(c_x, A_x, x)
    c_x .-= b_x
    mul!(c_u, A_u, u)
    c_u .-= b_u
end

function eq_constraint_state_jacobian!(constraint::LinearConstraint, ∂c∂x, x, u)
    ncx = length(constraint.b_x_eq)

    ∂c∂x .= 0.0
    ∂c∂x[1:ncx, :] .= constraint.A_x_eq
end

function eq_constraint_input_jacobian!(constraint::LinearConstraint, ∂c∂u, x, u)
    b_x, b_u = constraint.b_x_eq, constraint.b_u_eq
    ncx = length(b_x)
    ncu = length(b_u)

    ∂c∂u .= 0.0
    ∂c∂u[(ncx+1):(ncx+ncu), :] .= constraint.A_u_eq
end

function ineq_constraint!(constraint::LinearConstraint, c, x, u)
    A_x, A_u = constraint.A_x_ineq, constraint.A_u_ineq
    b_x, b_u = constraint.b_x_ineq, constraint.b_u_ineq
    ncx = length(b_x)
    ncu = length(b_u)

    c_x = @view c[1:ncx]
    c_u = @view c[(ncx+1):(ncx+ncu)]

    mul!(c_x, A_x, x)
    c_x .-= b_x
    mul!(c_u, A_u, u)
    c_u .-= b_u
end

function ineq_constraint_state_jacobian!(constraint::LinearConstraint, ∂c∂x, x, u)
    b_x, b_u = constraint.b_x_ineq, constraint.b_u_ineq
    ncx = length(b_x)
    ncu = length(b_u)

    ∂c∂x .= 0.0
    ∂c∂x[1:ncx, :] .= constraint.A_x_ineq
end

function ineq_constraint_input_jacobian!(constraint::LinearConstraint, ∂c∂u, x, u)
    b_x, b_u = constraint.b_x_ineq, constraint.b_u_ineq
    ncx = length(b_x)
    ncu = length(b_u)

    ∂c∂u .= 0.0
    ∂c∂u[(ncx+1):(ncx+ncu), :] .= constraint.A_u_ineq
end