(define (problem generated_problem)
  (:domain manipulation-base)

  (:objects
    pen - item
    source_pen - location
  )

  (:init
    (on pen source_pen)
    (clear pen)
    (reachable source_pen)
    (gripper-empty)
    (camera-aimed-at pen)
  )

  (:goal
    (holding pen)
  )
)