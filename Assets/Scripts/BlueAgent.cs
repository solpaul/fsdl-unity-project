using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;


public class BlueAgent : Agent
{
    Rigidbody rBody;
    public float forceMultiplier = 10;
    public Transform Target;

    [Tooltip("Speed to pitch up or down")]
    public float pitchSpeed = 100;

    [Tooltip("Speed to rotate around the up axis")]
    public float yawSpeed = 100f;

    [Tooltip("Speed to rotate around the forward axis")]
    public float rollSpeed = 100f;

    // Allows for a time penalty
    private float timePenalty;
    private float accumulatedTimePenalty = 0f; 

    // Allows for smoother pitch changes
    private float smoothPitchChange = 0f;

    // Allows for smoother yaw changes
    private float smoothYawChange = 0f;

    // // Allows for smoother roll changes
    // private float smoothRollChange = 0f;

    public bool trainingMode;

    void Start () {
        rBody = GetComponent<Rigidbody>();

        // If the agent's MaxStep > 0, use it to calculate the
        // accumulated time penalty. To be used for per-step rewards
        if (MaxStep > 0)
        {
            timePenalty = 1f / MaxStep;
        }
    }

    
    public override void OnEpisodeBegin()
    {
       // If the Agent fell, zero its momentum
        // if (this.transform.localPosition.y < 0)
        // {
        //     this.rBody.angularVelocity = Vector3.zero;
        //     this.rBody.velocity = Vector3.zero;
        //     this.transform.localPosition = new Vector3( 0, 0.5f, 0);
        // }

        // Zero out velocities so that movement stops before a new episode begins
        // this.rBody.angularVelocity = Vector3.zero;
        // this.rBody.velocity = Vector3.zero;

        // Move the target to a new spot
        Target.localPosition = new Vector3(Random.value * 8 - 4,
                                           Random.value * 8 + 1,
                                           Random.value * 8 - 4);

        // Target.localPosition = new Vector3(3, 3, 3);

        // Move the agent to a new spot
        // transform.localPosition = new Vector3(Random.value * 8 - 4,
        //                                       Random.value * 8 + 1,
        //                                       Random.value * 8 - 4);

        // transform.localPosition = new Vector3(3, 3, 0);

        // Reset the agent rotation
        // transform.rotation = Quaternion.identity;  
    }
    
    public override void CollectObservations(VectorSensor sensor)
    {
        // Agent positions (3 observations)
        sensor.AddObservation(this.transform.localPosition);

        // Target position
        // sensor.AddObservation(Target.localPosition);

        // Position of center of face
        // Vector3 facePosition = transform.localPosition + transform.forward * 0.5f;

        // Vector from center of face to target
        // Vector3 toTarget = Target.localPosition - facePosition;

        // Vector from cube center to target center
        // Vector3 toTarget = Target.localPosition - transform.localPosition;

        // convert to local coordinate system
        // Vector3 toTargetLocal = this.transform.InverseTransformDirection(toTarget);

        // Observe a normalized vector pointing to the target (3 observations)
        // sensor.AddObservation(toTargetLocal.normalized);

        // Agent velocity (3 observations)
        sensor.AddObservation(rBody.velocity);

        // Observe the agent's local rotation (4 observations)
        sensor.AddObservation(transform.localRotation.normalized);

        // Observe the distance from agent to target surface (1 observation)
        // sensor.AddObservation(toTarget.magnitude - 0.5f);

        // Observe the angular velocity of the agent (3 observations)
        sensor.AddObservation(rBody.angularVelocity);
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        // Actions
        // var dirForward = Vector3.zero;
        // var forward = actionBuffers.ContinuousActions[0];
        // dirForward = transform.forward * forward;
        // rBody.AddForce(dirForward * forceMultiplier);

        // Calculate movement vector
        Vector3 move = new Vector3(actionBuffers.ContinuousActions[0], 
                                   actionBuffers.ContinuousActions[1], 
                                   actionBuffers.ContinuousActions[2]);

        // Add force in the direction of the move vector
        rBody.AddForce(move * forceMultiplier);

        // Get the current rotation
        // Quaternion rotationVector = transform.localRotation;
        Vector3 rotationVector = transform.rotation.eulerAngles;

        float pitchChange = actionBuffers.ContinuousActions[3];
        float yawChange = actionBuffers.ContinuousActions[4];
        // float rollChange = actionBuffers.ContinuousActions[3];

        // Calculate smooth rotation changes
        smoothPitchChange = Mathf.MoveTowards(smoothPitchChange, pitchChange, 2f * Time.fixedDeltaTime);
        smoothYawChange = Mathf.MoveTowards(smoothYawChange, yawChange, 2f * Time.fixedDeltaTime);
        // smoothRollChange = Mathf.MoveTowards(smoothRollChange, rollChange, 2f * Time.fixedDeltaTime);

        // Calculate pitch, yaw and roll rotation
        // Quaternion rotate = Quaternion.Euler(smoothPitchChange * pitchSpeed * Time.fixedDeltaTime,
        //                                      smoothYawChange * yawSpeed * Time.fixedDeltaTime,0);
        // Quaternion yaw = Quaternion.Euler(0,smoothYawChange * yawSpeed * Time.fixedDeltaTime, 0);
        // Quaternion roll = Quaternion.Euler(0,0,smoothRollChange * rollSpeed * Time.fixedDeltaTime);

        // transform.localRotation = rotationVector * pitch * yaw * roll; 
        // transform.localRotation = rotationVector * rotate;

        // // Calculate new pitch and yaw based on smoothed values
        float pitch = rotationVector.x + smoothPitchChange * Time.fixedDeltaTime * pitchSpeed;
        float yaw = rotationVector.y + smoothYawChange * Time.fixedDeltaTime * yawSpeed;
        // float roll = rotationVector.z + smoothRollChange * Time.fixedDeltaTime * rollSpeed;

        // // Apply the new rotation
        transform.rotation = Quaternion.Euler(pitch, yaw, 0f);

        // // Rewards
        // float distanceToTarget = Vector3.Distance(this.transform.localPosition, Target.localPosition);

        // // Reached target
        // if (trainingMode && distanceToTarget < 1f)
        // {
        //     SetReward(1.0f);
        //     EndEpisode();
        // }

        // Fell off platform
        // else if (this.transform.localPosition.y < 0)
        // {
        //     EndEpisode();
        // }
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (trainingMode && collision.collider.CompareTag("floor"))
        {
            // Collided with the area boundary, give a negative reward
            AddReward(-.5f);
        }

        // Reached target
        if (trainingMode && collision.collider.CompareTag("target"))
        {

            // Get the the contact point
            ContactPoint contact = collision.GetContact(0);

            Vector3 localPoint = transform.InverseTransformPoint(contact.point);
            // Debug.Log(localPoint);

            // If z value is largest then it was the front face so reward and reset
            if (localPoint.z > Mathf.Abs(localPoint.x) && localPoint.z > Mathf.Abs(localPoint.y))
            // If x value is largest then it was the x-front face so reward and reset
            // if (localPoint.x > Mathf.Abs(localPoint.z) && localPoint.x > Mathf.Abs(localPoint.y))
            {
                accumulatedTimePenalty = timePenalty * StepCount;
                SetReward(1.0f - accumulatedTimePenalty);
                EndEpisode();
            }
            // else give a negative reward for touching anywhere else on the agent
            // else
            // {
            //     SetReward(-.1f);
            // }
        }
    }
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        // continuousActionsOut[0] = Input.GetAxis("Horizontal");
        // continuousActionsOut[1] = Input.GetAxis("Vertical");

        // Create placeholders for all movement/turning
        Vector3 forward = Vector3.zero;
        Vector3 left = Vector3.zero;
        Vector3 up = Vector3.zero;
        float pitch = 0f;
        float yaw = 0f;
        // float roll = 0f;

        // Convert keyboard inputs to movement and turning
        // All values should be between -1 and +1

        // Forward/backward
        if (Input.GetKey(KeyCode.W)) forward = transform.forward;
        else if (Input.GetKey(KeyCode.S)) forward = -transform.forward;

        // Left/right
        if (Input.GetKey(KeyCode.A)) left = -transform.right;
        else if (Input.GetKey(KeyCode.D)) left = transform.right;

        // Up/down
        if (Input.GetKey(KeyCode.E)) up = transform.up;
        else if (Input.GetKey(KeyCode.C)) up = -transform.up;

        // Combine the movement vectors and normalize
        Vector3 combined = (forward + left + up).normalized;

        // Pitch up/down
        if (Input.GetKey(KeyCode.UpArrow)) pitch = 1f;
        else if (Input.GetKey(KeyCode.DownArrow)) pitch = -1f;

        // Turn left/right
        if (Input.GetKey(KeyCode.LeftArrow)) yaw = -1f;
        else if (Input.GetKey(KeyCode.RightArrow)) yaw = 1f;

        // Roll left/right
        // if (Input.GetKey(KeyCode.Z)) roll = -1f;
        // else if (Input.GetKey(KeyCode.X)) roll = 1f;

        // Add the 3 movement values, pitch, yaw and roll to the actionsOut array
        continuousActionsOut[0] = combined.x;
        continuousActionsOut[1] = combined.y;
        continuousActionsOut[2] = combined.z;
        continuousActionsOut[3] = pitch;
        continuousActionsOut[4] = yaw;
        // continuousActionsOut[5] = roll;
    }
}

