(define (problem generated_problem)
  (:domain manipulation-base)

  (:objects
    pen - item
    keyboard - location
  )

  (:init
    (holding pen)
    (clear pen)
    (reachable keyboard)
  )

  (:goal
    (on pen keyboard)
  )
)