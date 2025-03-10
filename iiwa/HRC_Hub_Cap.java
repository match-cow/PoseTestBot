package application;

import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.net.SocketException;
import java.net.UnknownHostException;
import java.util.concurrent.TimeUnit;

import javax.inject.Inject;

import com.kuka.roboticsAPI.applicationModel.RoboticsAPIApplication;
import static com.kuka.roboticsAPI.motionModel.BasicMotions.*;

import com.kuka.roboticsAPI.deviceModel.LBR;
import com.kuka.roboticsAPI.geometricModel.AbstractFrame;
import com.kuka.roboticsAPI.geometricModel.Frame;
import com.kuka.roboticsAPI.geometricModel.ObjectFrame;
import com.kuka.roboticsAPI.persistenceModel.templateModel.InfoTemplate;
import com.kuka.roboticsAPI.motionModel.IMotionContainer;

import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;

/**
 * HRC_Hub_Cap is a KUKA Sunrise application that controls an LBR robot to
 * perform a series of motions and transmits the robot's pose data over UDP.
 * The application waits for a "start" signal via UDP before initiating the
 * motion sequence.
 */
public class HRC_Hub_Cap extends RoboticsAPIApplication {
	@Inject
	private LBR robot;
	@Inject
	private InfoTemplate robotinfo;

	private IMotionContainer motion;

	private int sampleTime = 10;
	private int CaptureVel;
	private int sleepTime = 500;

	private AbstractFrame Template_Base_ref;

	private String receiver_ip = "172.31.1.151";
	private int receiver_port = 8080;
	private int robot_port = 30300;
	private double ptpVel = 0.25;

	@Override
	public void initialize() {
		// initialize your application here
		robot = getContext().getDeviceFromType(LBR.class);
		robotinfo.setBase("/HRC_Hub/Template_Base");
		getLogger().info("getBase: " + robotinfo.getBase());
		getLogger().info("getTool: " + robotinfo.getTool());
		getLogger().info("getValue: " + robotinfo.getValue());
		Template_Base_ref = getApplicationData().getFrame(
				"/HRC_Hub/Template_Base");
	}

	@Override
	public void run() {

		// your application execution starts here
		ObjectFrame Template_Base = getApplicationData().getFrame(
				"/HRC_Hub/Template_Base");
		getLogger().info("Template_Base: " + Template_Base);

		Frame TestFrame = (new Frame(0, -300, 300, Math.toRadians(-90),
				Math.toRadians(0), Math.toRadians(180)));
		TestFrame.setParent(Template_Base);

		Frame Center = (new Frame(0, -320, 430, Math.toRadians(-90),
				Math.toRadians(30), Math.toRadians(180)));
		Center.setParent(Template_Base);

		Frame Right = (new Frame(420, -210, 540, Math.toRadians(-25),
				Math.toRadians(30), Math.toRadians(178.99)));
		Right.setParent(Template_Base);

		Frame Left = (new Frame(-420, -210, 540, Math.toRadians(-165),
				Math.toRadians(30), Math.toRadians(178.99)));
		Left.setParent(Template_Base);

		Frame CenterClose = (new Frame(0, -250, 350, Math.toRadians(-90),
				Math.toRadians(30), Math.toRadians(178.99)));
		CenterClose.setParent(Template_Base);

		Frame RightClose = (new Frame(230, -110, 350, Math.toRadians(-30),
				Math.toRadians(30), Math.toRadians(175.53)));
		RightClose.setParent(Template_Base);

		Frame LeftClose = (new Frame(-230, -110, 350, Math.toRadians(-170),
				Math.toRadians(30), Math.toRadians(175.53)));
		LeftClose.setParent(Template_Base);

		Frame Top = (new Frame(0, -320, 840, Math.toRadians(-90),
				Math.toRadians(18), Math.toRadians(180)));
		Top.setParent(Template_Base);

		Frame Bottom = (new Frame(0, -160, 350, Math.toRadians(-90),
				Math.toRadians(18), Math.toRadians(180)));
		Bottom.setParent(Template_Base);

		// Move to all poses ptp to test reachability
		// robot.move(ptp(Center));
		// robot.move(ptp(Left));
		//
		// robot.move(ptp(RightClose));
		// robot.move(ptp(CenterClose));
		// robot.move(ptp(LeftClose));
		//
		// robot.move(ptp(Top));
		// robot.move(ptp(Bottom));

		
		while (true) {
		
			while (true) {
				try {
				DatagramSocket socket = new DatagramSocket(robot_port);
				getLogger().info("Waiting for UDP message...");
				byte[] receiveData = new byte[1024];
				DatagramPacket receivePacket = new DatagramPacket(receiveData,
						receiveData.length);
				socket.receive(receivePacket);

				String jsonMessage = new String(receivePacket.getData(), 0,
						receivePacket.getLength());
				getLogger().info("Received message: " + jsonMessage);

				JSONParser parser = new JSONParser();
				JSONObject jsonObject = (JSONObject) parser.parse(jsonMessage);

				if (jsonObject.containsKey("start")) {
					Long startValue = (Long) jsonObject.get("start");
					getLogger().info("Start value: " + startValue);
					CaptureVel = startValue.intValue();
					// You can save or process the startValue here
					socket.close();
					break; // Exit the loop after processing the "start" key
				} else {
					getLogger().info("Waiting for message with 'start' key...");
				}}		
		catch (Exception e) {

		}}
		
				
				
		robot.move(ptp(Right).setJointVelocityRel(ptpVel));
		// Small delay before start of motion
		try {
			Thread.sleep(sleepTime);
		} catch (InterruptedException e) {
		}

		motion = robot
				.moveAsync(circ(Center, Left).setCartVelocity(CaptureVel));

		transmitpose(motion, sampleTime, "circ_far");

		robot.move(ptp(RightClose).setJointVelocityRel(ptpVel));

		try {
			Thread.sleep(sleepTime);
		} catch (InterruptedException e) {
		}

		motion = robot.moveAsync(circ(CenterClose, LeftClose).setCartVelocity(
				CaptureVel));

		transmitpose(motion, sampleTime, "circ_close");

		robot.move(ptp(Top).setJointVelocityRel(ptpVel));

		try {
			Thread.sleep(sleepTime);
		} catch (InterruptedException e) {
		}

		motion = robot.moveAsync(lin(Bottom).setCartVelocity(CaptureVel));

		transmitpose(motion, sampleTime, "zoom");

		motion = robot.moveAsync(ptp(Center).setJointVelocityRel(ptpVel));

		transmitpose(motion, 1000, "end");
		
		}

	}

	/**
	 * Transmits the robot's pose data over UDP during the motion execution.
	 *
	 * @param motion      The motion container being executed.
	 * @param sampleTime  The sampling interval in milliseconds.
	 * @param motion_name A descriptive name for the motion.
	 */
	@SuppressWarnings("unchecked")
	public void transmitpose(IMotionContainer motion, int sampleTime,
			String motion_name) {

		try {
			DatagramSocket socket = new DatagramSocket();

			InetAddress destAddress = InetAddress.getByName(receiver_ip);
			int destPort = receiver_port;

			while (!motion.isFinished()) {

				Frame currentPose = robot.getCurrentCartesianPosition(
						robot.getFlange(), Template_Base_ref);

				Double X = currentPose.getX();
				Double Y = currentPose.getY();
				Double Z = currentPose.getZ();
				Double A = currentPose.getAlphaRad();
				Double B = currentPose.getBetaRad();
				Double C = currentPose.getGammaRad();

				// Create JSON Array
				JSONObject jsonObject = new JSONObject();
				jsonObject.put("motion", motion_name);
				jsonObject.put("X", X);
				jsonObject.put("Y", Y);
				jsonObject.put("Z", Z);
				jsonObject.put("A", A);
				jsonObject.put("B", B);
				jsonObject.put("C", C);

				// Convert to JSON string
				String jsonString = jsonObject.toJSONString();

				String payload = jsonString;

				byte[] data = payload.getBytes();

				DatagramPacket packet = new DatagramPacket(data, data.length,
						destAddress, destPort);

				try {
					socket.send(packet);
					TimeUnit.MILLISECONDS.sleep(sampleTime);
				} catch (IOException e) {
				} catch (InterruptedException e) {
				}
			}
			synchronized (motion) {
				motion.notify();
			}

			socket.close();

		} catch (SocketException e) {
		} catch (UnknownHostException e) {
		}
	}
}
