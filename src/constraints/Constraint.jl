
abstract type Constraint{T} end

nc(constraint::Constraint)::Int = nc_eq(constraint) + nc_ineq(constraint)
nc_eq(constraint::Constraint)::Int = 0
nc_ineq(constraint::Constraint)::Int = 0

function constraint!(constraint::Constraint, c, x, u)
    c_eq = @view c[1:nc_eq(constraint)]
    c_ineq = @view c[(nc_eq(constraint)+1):end]

    eq_constraint!(constraint, c_eq, x, u)
    ineq_constraint!(constraint, c_ineq, x, u)
end

function constraint_state_jacobian!(constraint::Constraint, ∂c∂x, x, u)
    cx_eq = @view ∂c∂x[1:nc_eq(constraint), :]
    cx_ineq = @view ∂c∂x[(nc_eq(constraint)+1):end, :]

    eq_constraint_state_jacobian!(constraint, cx_eq, x, u)
    ineq_constraint_state_jacobian!(constraint, cx_ineq, x, u)
end

function constraint_input_jacobian!(constraint::Constraint, ∂c∂u, x, u)
    cu_eq = @view ∂c∂u[1:nc_eq(constraint), :]
    cu_ineq = @view ∂c∂u[(nc_eq(constraint)+1):end, :]

    eq_constraint_input_jacobian!(constraint, cu_eq, x, u)
    ineq_constraint_input_jacobian!(constraint, cu_ineq, x, u)
end

function eq_constraint!(constraint::Constraint, c, x, u)
    c .= 0
end

function eq_constraint_state_jacobian!(constraint::Constraint, ∂c∂x, x, u)
    ForwardDiff.jacobian!(∂c∂x, (c, x) -> constraint!(constraint, c, x, u), x)
end

function eq_constraint_input_jacobian!(constraint::Constraint, ∂c∂u, x, u)
    ForwardDiff.jacobian!(∂c∂u, (c, u) -> constraint!(constraint, c, x, u), u)
end

function ineq_constraint!(constraint::Constraint, c, x, u)
    c .= 0
end

function ineq_constraint_state_jacobian!(constraint::Constraint, ∂c∂x, x, u)
    ForwardDiff.jacobian!(∂c∂x, (c, x) -> constraint!(constraint, c, x, u), x)
end

function ineq_constraint_input_jacobian!(constraint::Constraint, ∂c∂u, x, u)
    ForwardDiff.jacobian!(∂c∂u, (c, u) -> constraint!(constraint, c, x, u), u)
end

include("linear_constraint.jl")